"""
UVLM_Qwen3: Report generation network using Qwen3-4B as LLM

Maintains the same structure as the lightweight LLM version, only replacing the LLM part with Qwen3-4B + LoRA.
Does not support deepstack functionality.
"""

from typing import Tuple, Union, List, Optional, Type
import torch
import torch.nn as nn
import math
import os

from transformers import AutoModelForCausalLM, AutoTokenizer

# Import all components from the base class
from uvlm.networks.uvlm import (
    ResidualEncoder,
    ClassificationHead,
    AdaptivePoolEmbed,
    PatchEmbed,
)
from dynamic_network_architectures.building_blocks.residual import BasicBlockD


class Qwen3LLMReportGenerator(nn.Module):
    """
    Report generator using Qwen3-4B as LLM

    Maintains the same interface as the lightweight LLM version, only replacing the internal LLM with Qwen3 + LoRA.
    Directly reuses Qwen3's embedding weights.
    """

    def __init__(
        self,
        llm_model_path: str,
        llm_embed_dim: int = 512,  # Dimension after visual feature projection, consistent with lightweight version
        max_length: int = 8192,
        device: Optional[torch.device] = None,
        use_lora: bool = True,
        lora_r: int = 64,
        lora_alpha: int = 64,  # Default alpha = r to avoid gradient explosion
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        vision_token_buffer: int = 512,
    ):
        super().__init__()
        self.llm_embed_dim = llm_embed_dim
        self.max_length = max_length
        self.use_lora = use_lora
        self.llm_model_path = llm_model_path
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit

        # Load tokenizer
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_model_path,
            trust_remote_code=True,
            use_fast=True,
            model_max_length=max_length,
            padding_side="right",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Add vision special tokens (consistent with lightweight version)
        special_tokens = ["<|vision_start|>", "<|vision_end|>", "<|vision_pad|>"]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        self.vision_start_token_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self.vision_end_token_id = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        self.vision_pad_token_id = self.tokenizer.convert_tokens_to_ids("<|vision_pad|>")

        # Load LLM
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                llm_model_path,
                trust_remote_code=True,
                quantization_config=quantization_config,
                device_map="auto",
            )
        elif load_in_8bit:
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                llm_model_path,
                trust_remote_code=True,
                load_in_8bit=True,
                device_map="auto",
            )
        else:
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                llm_model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )

        # Get LLM's hidden_size
        self.model_hidden_size = self.llm_model.config.hidden_size

        # Adjust embedding size to accommodate new special tokens
        # New tokens will use default initialization (close to existing embedding distribution)
        self.llm_model.resize_token_embeddings(len(self.tokenizer))

        # Apply LoRA
        if use_lora:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

            if load_in_8bit or load_in_4bit:
                self.llm_model = prepare_model_for_kbit_training(self.llm_model)

            # Default LoRA target modules
            if lora_target_modules is None:
                lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.llm_model = get_peft_model(self.llm_model, lora_config)
            self.llm_model.print_trainable_parameters()

        # Vision projection: llm_embed_dim -> model_hidden_size
        # Consistent structure with lightweight version, only target dimension differs
        self.vision_proj = nn.Sequential(
            nn.Linear(llm_embed_dim, self.model_hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(self.model_hidden_size, self.model_hidden_size, bias=True),
        )
        # Initialization
        for module in self.vision_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        self.vision_norm = nn.LayerNorm(self.model_hidden_size, eps=1e-6)
        self.eos_token_id = self.tokenizer.eos_token_id
        self._debug_printed = False

    @property
    def device(self):
        """Dynamically get device"""
        return next(self.parameters()).device

    def get_input_embeddings(self):
        """Get LLM's input embeddings"""
        if hasattr(self.llm_model, 'get_input_embeddings'):
            return self.llm_model.get_input_embeddings()
        elif hasattr(self.llm_model, 'model') and hasattr(self.llm_model.model, 'embed_tokens'):
            return self.llm_model.model.embed_tokens
        else:
            raise AttributeError("Cannot find input embeddings in LLM model")

    def forward(
        self,
        vision_feature: torch.Tensor,
        text_prompt: Optional[str] = None,
        text_ids: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None,
        generate: bool = False,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        deepstack_features: Optional[List[torch.Tensor]] = None,  # Keep interface compatible, but not used
        deepstack_vision_lengths: Optional[List[int]] = None,  # Keep interface compatible, but not used
        skip_special_tokens: bool = True,
    ):
        if generate:
            return self._generate(
                vision_feature, text_prompt, max_new_tokens,
                temperature, top_p, top_k, skip_special_tokens
            )
        else:
            return self._train_forward(vision_feature, text_ids, text_attention_mask)

    def _train_forward(self, vision_feature, text_ids, text_attention_mask):
        """Training forward propagation"""
        B = vision_feature.shape[0]
        device = vision_feature.device

        # Get text embeddings
        embed_tokens = self.get_input_embeddings()
        text_embeds = embed_tokens(text_ids)
        # Get visual special token embeddings
        vs_embed = embed_tokens(
            torch.tensor([[self.vision_start_token_id]], device=device)
        ).expand(B, 1, -1)
        ve_embed = embed_tokens(
            torch.tensor([[self.vision_end_token_id]], device=device)
        ).expand(B, 1, -1)

        # Project visual features to LLM hidden_size
        vision_feature = self.vision_proj(vision_feature)
        vision_feature = self.vision_norm(vision_feature)

        # Ensure dtype consistency
        target_dtype = text_embeds.dtype
        vision_feature = vision_feature.to(dtype=target_dtype)

        # Concatenate: [vision_start] + [vision_tokens] + [vision_end] + [text_tokens]
        inputs_embeds = torch.cat([vs_embed, vision_feature, ve_embed, text_embeds], dim=1)

        # Build attention mask
        vision_len = 2 + vision_feature.shape[1]
        vision_attn = torch.ones(B, vision_len, device=device, dtype=text_attention_mask.dtype)
        attention_mask = torch.cat([vision_attn, text_attention_mask], dim=1)

        # Debug print (only once)
        if not self._debug_printed:
            self._debug_printed = True
            print(f"[Qwen3 LLM DEBUG] Sequence length breakdown:")
            print(f"  - special tokens: 2 (vision_start + vision_end)")
            print(f"  - visual tokens: {vision_feature.shape[1]}")
            print(f"  - text tokens: {text_embeds.shape[1]}")
            print(f"  - total seq_len: {inputs_embeds.shape[1]}")

        # LLM forward pass
        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
        )

        return {'llm_logits': outputs.logits, 'text_start_idx': vision_len}

    def _generate(
        self,
        vision_feature,
        text_prompt,
        max_new_tokens,
        temperature,
        top_p,
        top_k,
        skip_special_tokens,
    ):
        """Generate report"""
        B = vision_feature.shape[0]
        device = vision_feature.device

        # Tokenize prompt
        prompt_inputs = self.tokenizer(
            text_prompt or "",
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length // 2,
        ).to(device)

        # Get embeddings
        embed_tokens = self.get_input_embeddings()
        text_embeds = embed_tokens(prompt_inputs.input_ids)

        if text_embeds.shape[0] != B:
            text_embeds = text_embeds.expand(B, -1, -1)
            prompt_attention_mask = prompt_inputs.attention_mask.expand(B, -1)
        else:
            prompt_attention_mask = prompt_inputs.attention_mask

        # Get visual special token embeddings
        vs_embed = embed_tokens(
            torch.tensor([[self.vision_start_token_id]], device=device)
        ).expand(B, 1, -1)
        ve_embed = embed_tokens(
            torch.tensor([[self.vision_end_token_id]], device=device)
        ).expand(B, 1, -1)

        # Project visual features
        vision_feature = self.vision_proj(vision_feature)
        vision_feature = self.vision_norm(vision_feature)

        # Ensure dtype consistency
        target_dtype = text_embeds.dtype
        vision_feature = vision_feature.to(dtype=target_dtype)

        # Concatenate
        inputs_embeds = torch.cat([vs_embed, vision_feature, ve_embed, text_embeds], dim=1)

        # Build attention mask
        vision_len = 2 + vision_feature.shape[1]
        vision_attn = torch.ones(B, vision_len, device=device, dtype=prompt_attention_mask.dtype)
        attention_mask = torch.cat([vision_attn, prompt_attention_mask], dim=1)

        # Generate
        with torch.no_grad():
            outputs = self.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                eos_token_id=self.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True,
                use_cache=True,
            )

        # Decode generated tokens
        # When using inputs_embeds, the sequence returned by generate() directly contains the generated tokens
        # No need to slice from input_seq_len position, decode the entire output directly
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=skip_special_tokens)

        return generated_texts


class UVLM_Qwen3(nn.Module):
    """
    Report generation network using Qwen3-4B as LLM

    Maintains the same structure as the lightweight LLM version, only replacing the LLM part.
    Does not support deepstack functionality.
    """

    def __init__(
        self,
        in_channels: int,
        patch_kernel_sizes: List[Union[int, Tuple[int, int, int]]],
        enable_report_gen: bool = True,
        # Qwen3 LLM parameters
        llm_model_path: str = "/path/to/model/",
        llm_embed_dim: int = 512,
        report_max_length: int = 8192,
        use_lora: bool = True,
        lora_r: int = 64,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        # Generation parameters
        generation_temperature: float = 0.7,
        generation_top_p: float = 0.9,
        generation_top_k: int = 50,
        # Encoder parameters
        n_stages: int = 6,
        features_per_stage: List[int] = [16, 32, 64, 128, 256, 320],
        kernel_sizes: List[Union[Tuple[int, int, int], int]] = None,
        strides: List[Union[Tuple[int, int, int], int]] = None,
        n_blocks_per_stage: List[int] = None,
        norm_op: nn.Module = nn.InstanceNorm3d,
        norm_op_kwargs: dict = None,
        conv_op: nn.Module = nn.Conv3d,
        conv_bias: bool = True,
        nonlin: nn.Module = nn.LeakyReLU,
        nonlin_kwargs: dict = None,
        dropout_op=None,
        dropout_op_kwargs=None,
        use_stages: Optional[List[int]] = None,
        device: Optional[torch.device] = None,
        # Classification parameters
        only_cls: bool = False,
        cls_head_num_classes_list: List[int] = [1, 13],
        cls_drop_out_list: List[float] = [0.0, 0.0],
        cls_query_num_list: List[int] = [2, 16],
        use_cross_attention: bool = True,
        # Pool parameters
        use_adaptive_pool: bool = True,
        pool_output_size: Tuple[int, int, int] = None,
        pool_output_size_list: List[Tuple[int, int, int]] = None,
        visual_token_length_source_stage: int = -1,
        # Mode parameters
        mode: str = 'both',
        # Compatibility parameters (keep interface but do not use deepstack)
        num_heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.0,
        use_weight_tying: bool = True,
        use_deepstack: bool = False,  # Not used, kept for interface compatibility
        use_vision_aware_mask: bool = True,
        deepstack_skip_stages: int = 0,
        layers_per_stage: int = 1,
        use_gate: bool = False,
        tokenizer_path: str = None,
        report_max_new_tokens: int = 1024,
    ):
        super().__init__()

        # Mode settings
        self.mode = mode
        self.enable_cls = mode in ['only_cls', 'both']
        self.enable_report_gen = mode in ['only_report', 'both']
        self.only_cls = (mode == 'only_cls')
        # Save generation parameters
        self.generation_temperature = generation_temperature
        self.generation_top_p = generation_top_p
        self.generation_top_k = generation_top_k

        self.n_stages = n_stages
        self.use_stages = use_stages or list(range(n_stages))
        self.pool_output_size = pool_output_size
        self.pool_output_size_list = pool_output_size_list
        self.visual_token_length_source_stage = visual_token_length_source_stage
        self.use_adaptive_pool = use_adaptive_pool

        # Build Encoder (consistent with lightweight version)
        self.encoder = ResidualEncoder(
            input_channels=in_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=conv_op,
            kernel_sizes=kernel_sizes,
            strides=strides,
            n_blocks_per_stage=n_blocks_per_stage,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            block=BasicBlockD,
            return_skips=True,
            stem_channels=features_per_stage[0],
        )

        final_features = features_per_stage[self.use_stages[-1]]

        # Classification head (consistent with lightweight version)
        self.cls_head_list = nn.ModuleList()
        if self.enable_cls:
            self.cls_drop_out_list = cls_drop_out_list
            self.cls_query_num_list = cls_query_num_list

            for i in range(len(cls_head_num_classes_list)):
                cls_head_num_classes = cls_head_num_classes_list[i]
                cls_drop_out = self.cls_drop_out_list[i]
                cls_query_num = self.cls_query_num_list[i]

                self.cls_head_list.append(ClassificationHead(
                    embed_dim=final_features,
                    query_num=cls_query_num,
                    num_classes=cls_head_num_classes,
                    dropout=cls_drop_out,
                    use_cross_attention=use_cross_attention,
                    num_heads=4
                ))

        # Report generation module
        if self.enable_report_gen:
            # Determine source stage index (consistent with lightweight version)
            if visual_token_length_source_stage < 0:
                self.source_stage_idx = n_stages + visual_token_length_source_stage
            else:
                self.source_stage_idx = visual_token_length_source_stage
            self.source_stage_idx = max(0, min(self.source_stage_idx, n_stages - 1))

            source_features = features_per_stage[self.source_stage_idx]

            # Pool embedding (consistent with lightweight version)
            if pool_output_size_list is not None:
                stage_pool_size = pool_output_size_list[self.source_stage_idx] if self.source_stage_idx < len(pool_output_size_list) else None
            else:
                stage_pool_size = pool_output_size

            if stage_pool_size is None:
                self.pool_embed = None
            elif use_adaptive_pool:
                self.pool_embed = AdaptivePoolEmbed(stage_pool_size)
            else:
                self.pool_embed = PatchEmbed(
                    source_features, source_features,
                    patch_kernel_sizes[self.source_stage_idx] if self.source_stage_idx < len(patch_kernel_sizes) else (2, 2, 2),
                    bias=True
                )

            self.pool_norm = nn.LayerNorm(source_features)

            # Vision projection: source_features -> llm_embed_dim (consistent with lightweight version)
            self.vision_proj = nn.Sequential(
                nn.Linear(source_features, llm_embed_dim * 2, bias=True),
                nn.GELU(),
                nn.Linear(llm_embed_dim * 2, llm_embed_dim, bias=True),
            )
            for module in self.vision_proj:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
            self.vision_norm = nn.LayerNorm(llm_embed_dim, eps=1e-6)

            # Calculate vision_token_buffer
            vision_token_buffer = 512
            if pool_output_size is not None:
                expected_vision_tokens = pool_output_size[0] * pool_output_size[1] * pool_output_size[2]
                vision_token_buffer = max(512, expected_vision_tokens + 128)
                print(f"[ResEncoderUVLM_Qwen3] pool_output_size={pool_output_size}, "
                      f"expected_vision_tokens={expected_vision_tokens}, "
                      f"vision_token_buffer={vision_token_buffer}")

            # Qwen3 LLM (only replace this part)
            self.llm_report_gen = Qwen3LLMReportGenerator(
                llm_model_path=llm_model_path,
                llm_embed_dim=llm_embed_dim,
                max_length=report_max_length,
                use_lora=use_lora,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_target_modules=lora_target_modules,
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                vision_token_buffer=vision_token_buffer,
            )

    @property
    def device(self):
        """Dynamically get device"""
        return next(self.parameters()).device

    def forward(
        self,
        input_image,
        generate_report=False,
        report_prompt=None,
        report_text_ids=None,
        report_text_attention_mask=None,
        max_new_tokens: int = 512,
        skip_special_tokens: bool = True,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
    ):
        # Encoder forward pass (consistent with lightweight version)
        all_stage_features = self.encoder(input_image, return_all_stages=True)
        x = all_stage_features[-1]

        # Classification (consistent with lightweight version)
        cls_pred_list = []
        if self.enable_cls:
            for cls_head in self.cls_head_list:
                cls_pred_list.append(cls_head(x))

        if not self.enable_report_gen:
            return None, cls_pred_list

        # Get visual features (consistent with lightweight version)
        stage_feat = all_stage_features[self.source_stage_idx]

        # Pool (if needed)
        if self.pool_embed is None:
            vision_tokens = stage_feat.flatten(2).permute(0, 2, 1)
        else:
            vision_tokens = self.pool_embed(stage_feat)

        # Vision projection (consistent with lightweight version)
        vision_tokens = self.pool_norm(vision_tokens)
        vision_tokens = self.vision_proj(vision_tokens)
        vision_tokens = self.vision_norm(vision_tokens)

        # Generation parameters
        use_temperature = temperature if temperature is not None else self.generation_temperature
        use_top_p = top_p if top_p is not None else self.generation_top_p
        use_top_k = top_k if top_k is not None else self.generation_top_k

        if generate_report:
            report_output = self.llm_report_gen(
                vision_tokens,
                text_prompt=report_prompt,
                generate=True,
                max_new_tokens=max_new_tokens,
                temperature=use_temperature,
                top_p=use_top_p,
                top_k=use_top_k,
                skip_special_tokens=skip_special_tokens,
            )
        else:
            report_output = self.llm_report_gen(
                vision_tokens,
                text_ids=report_text_ids,
                text_attention_mask=report_text_attention_mask,
                generate=False,
            )

        return report_output, cls_pred_list
