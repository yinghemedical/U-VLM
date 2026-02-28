from typing import Tuple, Union, List, Optional, Type
import torch
import torch.nn as nn
import math
from dynamic_network_architectures.building_blocks.residual import StackedResidualBlocks, BasicBlockD
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks


# ==================== ResidualEncoder ====================
class ResidualEncoder(nn.Module):
    """
    Residual Encoder module.

    Structure:
    - stem: StackedConvBlocks (1 convolutional layer, mapping input_channels to stem_channels)
    - stages: nn.Sequential of StackedResidualBlocks (one per stage)

    Weight naming:
    - stem.xxx
    - stages.0.xxx, stages.1.xxx, ...
    """
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: List[int],
        conv_op: Type[nn.Module],
        kernel_sizes: List[Union[Tuple[int, int, int], int]],
        strides: List[Union[Tuple[int, int, int], int]],
        n_blocks_per_stage: List[int],
        conv_bias: bool = True,
        norm_op: Type[nn.Module] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Type[nn.Module] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Type[nn.Module] = None,
        nonlin_kwargs: dict = None,
        block: Type[nn.Module] = BasicBlockD,
        return_skips: bool = False,
        stem_channels: int = None,
    ):
        super().__init__()

        self.return_skips = return_skips
        self.n_stages = n_stages

        # Stem: Use StackedConvBlocks, no stride/pooling, only channel mapping
        if stem_channels is None:
            stem_channels = features_per_stage[0]

        self.stem = StackedConvBlocks(
            1,  # num_convs
            conv_op,
            input_channels,
            stem_channels,
            kernel_sizes[0],
            1,  # stride=1, stem does not perform downsampling
            conv_bias,
            norm_op,
            norm_op_kwargs,
            dropout_op,
            dropout_op_kwargs,
            nonlin,
            nonlin_kwargs
        )

        # Stages: Use nn.Sequential to wrap StackedResidualBlocks
        stages = []
        current_input_channels = stem_channels

        for s in range(n_stages):
            stage = StackedResidualBlocks(
                n_blocks_per_stage[s],
                conv_op,
                current_input_channels,
                features_per_stage[s],
                kernel_sizes[s],
                strides[s],
                conv_bias,
                norm_op,
                norm_op_kwargs,
                dropout_op,
                dropout_op_kwargs,
                nonlin,
                nonlin_kwargs,
                block=block
            )
            stages.append(stage)
            current_input_channels = features_per_stage[s]

        self.stages = nn.Sequential(*stages)
        self.output_channels = features_per_stage
        self.strides = strides

    def forward(self, x, return_all_stages: bool = False):
        """
        Args:
            x: Input tensor
            return_all_stages: Whether to return outputs from all stages (for deepstack)

        Returns:
            If return_all_stages=True: List[Tensor], output from each stage
            If return_all_stages=False and return_skips=True: List[Tensor], output from each stage
            Otherwise: Output from the last stage
        """
        x = self.stem(x)

        if return_all_stages or self.return_skips:
            ret = []
            for s in self.stages:
                x = s(x)
                ret.append(x)
            return ret
        else:
            x = self.stages(x)
            return x


# ==================== Classification Head ====================
class CrossAttentionPooling(nn.Module):
    def __init__(self, embed_dim, query_num, num_classes, num_heads=4, dropout=0.0):
        super(CrossAttentionPooling, self).__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.query_num = query_num

        # Learnable query vectors, shape [query_num, embed_dim]
        self.class_query = nn.Parameter(torch.randn(query_num, embed_dim))

        # Cross Attention layer
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False
        )

        # LayerNorm and Dropout
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Classifier: Map [query_num * D] to [num_classes]
        # Note: query_num * embed_dim here
        self.classifier = nn.Linear(query_num * embed_dim, num_classes)

        # Initialization
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.class_query)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)

    def forward(self, x):
        """
        Args:
            x: Image features [B, D, H, W, L] or [B, D, H*W*L]

        Returns:
            Classification logits [B, num_classes]
        """
        batch_size = x.shape[0]

        # Process input features
        if x.dim() == 5:  # [B, D, H, W, L]
            x = x.flatten(2)  # [B, D, H*W*L]

        # Adjust dimensions: [B, D, L] -> [L, B, D] (seq_len, batch, embed_dim)
        x = x.permute(2, 0, 1)  # [H*W*L, B, D]

        # Expand query vectors: [query_num, embed_dim] -> [query_num, B, D]
        query = self.class_query.unsqueeze(1).repeat(1, batch_size, 1)  # [query_num, B, D]

        # Cross Attention
        attended, attention_weights = self.cross_attention(
            query=query,
            key=x,
            value=x,
        )

        # attended shape: [query_num, B, D]
        attended = self.norm(attended)
        attended = self.dropout(attended)

        # **************************** Key Modification ****************************
        # Adjust dimensions: [query_num, B, D] -> [B, query_num, D]
        attended_permuted = attended.permute(1, 0, 2)

        # Flatten query and feature dimensions: [B, query_num, D] -> [B, query_num * D]
        # This preserves information learned by different queries
        attended_flatten = attended_permuted.flatten(1)

        # Apply classifier: Map [B, query_num * D] to [B, num_classes]
        logits = self.classifier(attended_flatten)  # [B, num_classes]

        return logits

class ClassificationHead(nn.Module):
    def __init__(
        self,
        embed_dim,
        query_num,
        num_classes,
        dropout=0.0,
        use_cross_attention=True,
        num_heads=4
    ):
        """
        Args:
            embed_dim (int): Embedding dimension
            num_classes (int): Number of classes
            dropout (float): Dropout rate
            use_cross_attention (bool): Whether to use cross attention pooling
            num_heads (int): Number of attention heads
        """
        super(ClassificationHead, self).__init__()

        if use_cross_attention:
            self.pooling = CrossAttentionPooling(
                embed_dim=embed_dim,
                query_num=query_num,
                num_classes=num_classes,
                num_heads=num_heads,
                dropout=dropout
            )
        else:
            self.pooling = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),  # Global average pooling
                nn.Flatten(1),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, num_classes)
            )

    def forward(self, x):
        return self.pooling(x)


# ==================== RoPE ====================
class RotaryEmbedding(nn.Module):
    """Standard RoPE for text tokens."""
    def __init__(self, dim: int, max_seq_len: int = 8192, base: int = 10000):
        super().__init__()
        assert dim % 2 == 0, f"RoPE requires even head_dim, got {dim}"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        # Use persistent=False so these buffers will be converted with model.to(dtype)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.max_seq_len = max_seq_len

    def forward(self, seq_len: int, device: torch.device):
        if seq_len > self.max_seq_len:
            print(f"[RoPE DEBUG] seq_len={seq_len} > max_seq_len={self.max_seq_len}")
            print(f"[RoPE DEBUG] Exceeds {seq_len - self.max_seq_len} tokens")
            print(f"[RoPE DEBUG] Suggestion: Increase report_max_length or reduce visual tokens")
        assert seq_len <= self.max_seq_len, f"seq_len({seq_len}) > max_seq_len({self.max_seq_len})"
        # Buffers automatically have the correct dtype from model.to()
        return self.cos_cached[:seq_len].to(device), self.sin_cached[:seq_len].to(device)

def apply_3d_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply 3D RoPE to vision tokens: x [B, N, D], cos/sin [N, D] -> [B, N, D]."""
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    
    cos_even = cos[..., 0::2].unsqueeze(0)
    cos_odd = cos[..., 1::2].unsqueeze(0)
    sin_even = sin[..., 0::2].unsqueeze(0)
    sin_odd = sin[..., 1::2].unsqueeze(0)
    
    x_rot_even = x_even * cos_even - x_odd * sin_even
    x_rot_odd = x_even * sin_odd + x_odd * cos_odd
    
    x_rot = torch.stack([x_rot_even, x_rot_odd], dim=-1).flatten(-2)
    return x_rot


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply RoPE to Q/K: [B, num_heads, seq_len, head_dim]."""
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    
    q_even, q_odd = q[..., 0::2], q[..., 1::2]
    k_even, k_odd = k[..., 0::2], k[..., 1::2]
    cos_even, sin_even = cos[..., 0::2], sin[..., 0::2]
    cos_odd, sin_odd = cos[..., 1::2], sin[..., 1::2]
    
    q_rot_even = q_even * cos_even - q_odd * sin_even
    q_rot_odd = q_even * sin_odd + q_odd * cos_odd
    k_rot_even = k_even * cos_even - k_odd * sin_even
    k_rot_odd = k_even * sin_odd + k_odd * cos_odd
    
    q_rot = torch.stack((q_rot_even, q_rot_odd), dim=-1).flatten(-2)
    k_rot = torch.stack((k_rot_even, k_rot_odd), dim=-1).flatten(-2)
    return q_rot, k_rot


# ==================== Transformer Decoder Layer ====================
class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffn_dim: int, dropout: float = 0.0,
                 max_seq_len: int = 8192, use_deepstack: bool = False, use_gate: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_deepstack = use_deepstack
        self.use_gate = use_gate

        self.rope = RotaryEmbedding(self.head_dim, max_seq_len)
        self.self_attn_norm = nn.LayerNorm(embed_dim)
        self.ffn_norm = nn.LayerNorm(embed_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )

        # Gate mechanism for DeepStack injection
        if use_deepstack:
            if use_gate:
                # Learnable gate: dynamically adjusts injection strength based on current hidden state
                self.gate_proj = nn.Sequential(
                    nn.Linear(embed_dim, embed_dim // 4),
                    nn.GELU(),
                    nn.Linear(embed_dim // 4, embed_dim),
                    nn.Sigmoid()
                )
                # Initialize gate to small values for weaker injection during early training
                nn.init.zeros_(self.gate_proj[-2].bias)
                nn.init.normal_(self.gate_proj[-2].weight, std=0.01)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                visual_hidden: Optional[torch.Tensor] = None,
                visual_pos_masks: Optional[torch.Tensor] = None):
        B, seq_len, _ = x.shape
        cos, sin = self.rope(seq_len, x.device)

        # Self-Attention
        resid = x
        x_norm = self.self_attn_norm(x)
        q = self.q_proj(x_norm).view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            # Ensure mask has same dtype as attn_scores
            attn_mask = attn_mask.to(dtype=attn_scores.dtype, device=attn_scores.device)
            attn_scores = attn_scores + attn_mask
        if key_padding_mask is not None:
            attn_scores = attn_scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
            )

        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = attn_weights @ v
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, seq_len, -1)
        attn_output = self.out_proj(attn_output)
        x = resid + self.dropout(attn_output)

        # DeepStack injection with gate mechanism
        if self.use_deepstack and visual_hidden is not None and visual_pos_masks is not None:
            # Ensure visual_hidden matches x's dtype
            visual_hidden = visual_hidden.to(dtype=x.dtype, device=x.device)
            visual_pos_masks = visual_pos_masks.bool()

            for b in range(B):
                positions = visual_pos_masks[b].nonzero(as_tuple=False).view(-1)
                if positions.numel() > 0:
                    num_to_use = min(positions.numel(), visual_hidden.shape[1])

                    if self.use_gate:
                        # Gated injection: compute gate values based on hidden state at current positions
                        current_hidden = x[b, positions[:num_to_use]]  # [num_to_use, embed_dim]
                        gate_values = self.gate_proj(current_hidden)  # [num_to_use, embed_dim]
                        x[b, positions[:num_to_use]] = x[b, positions[:num_to_use]] + gate_values * visual_hidden[b, :num_to_use]
                    else:
                        # Simple scaled injection
                        x[b, positions[:num_to_use]] = x[b, positions[:num_to_use]] + visual_hidden[b, :num_to_use]
        # FFN
        resid = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = resid + self.dropout(x)
        return x


# ==================== Simple Transformer Decoder ====================
class SimpleTransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 512, num_layers: int = 1,
                 num_heads: int = 8, ffn_dim: int = 2048, max_seq_length: int = 8192,
                 dropout: float = 0.0, use_weight_tying: bool = True, use_deepstack: bool = False,
                 use_vision_aware_mask: bool = True,
                 vision_bidirectional: bool = True,  # New: vision tokens use bidirectional attention
                 layers_per_stage: int = 1,          # New: number of layers per stage
                 use_gate: bool = True):             # New: whether to use gating mechanism
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.use_deepstack = use_deepstack
        self.use_vision_aware_mask = use_vision_aware_mask
        self.vision_bidirectional = vision_bidirectional  # Key improvement
        self.layers_per_stage = layers_per_stage
        self.use_gate = use_gate

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, num_heads, ffn_dim, dropout, max_seq_length, use_deepstack, use_gate)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        if use_weight_tying:
            self.lm_head.weight = self.token_embedding.weight

        self.dropout = nn.Dropout(dropout)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, input_ids: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                visual_pos_masks: Optional[torch.Tensor] = None,
                deepstack_visual_embeds: Optional[List[torch.Tensor]] = None,
                deepstack_vision_lengths: Optional[List[int]] = None):

        if input_ids is not None:
            B, seq_len = input_ids.shape
            x = self.token_embedding(input_ids)
        elif inputs_embeds is not None:
            B, seq_len, _ = inputs_embeds.shape
            x = inputs_embeds
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        x = self.dropout(x)

        # Get main vision token length (for base mask)
        vision_len = 0
        if visual_pos_masks is not None:
            vision_mask = visual_pos_masks[0] if visual_pos_masks.dim() > 1 else visual_pos_masks
            vision_len = vision_mask.sum().item()

        # Precompute base causal mask
        base_causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=x.device, dtype=x.dtype),
            diagonal=1
        )

        # Vision-aware mask: vision tokens use bidirectional attention
        # Note: Through padding scheme, all deepstack_features have been unified to the same length (vision_len)
        # Therefore, only a unified causal_mask is needed
        stage_causal_masks = None
        if visual_pos_masks is not None and self.vision_bidirectional and vision_len > 0:
            base_causal_mask = self._create_vision_aware_mask(seq_len, vision_len, x.device, x.dtype)
        elif self.use_vision_aware_mask and visual_pos_masks is not None:
            # Original logic (backward compatibility)
            vision_mask = visual_pos_masks[0]
            vision_indices = torch.where(vision_mask)[0]
            if len(vision_indices) > 0:
                vision_idx_grid = vision_indices.unsqueeze(0).expand(len(vision_indices), -1)
                vision_idx_grid_t = vision_indices.unsqueeze(1).expand(-1, len(vision_indices))
                base_causal_mask[vision_idx_grid_t, vision_idx_grid] = 0.0

        key_padding_mask = None if attention_mask is None else (attention_mask == 0)

        for layer_idx, layer in enumerate(self.layers):
            current_visual = None
            current_visual_pos_masks = visual_pos_masks

            if self.use_deepstack and deepstack_visual_embeds is not None:
                # Each stage corresponds to layers_per_stage layers
                stage_idx = layer_idx // self.layers_per_stage
                if stage_idx < len(deepstack_visual_embeds):
                    current_visual = deepstack_visual_embeds[stage_idx]
                    # Note: Through padding scheme, all deepstack_features have been unified to the same length
                    # Therefore, directly use visual_pos_masks without special handling

            # Select corresponding causal_mask
            if stage_causal_masks is not None:
                stage_idx = layer_idx // self.layers_per_stage
                if stage_idx < len(stage_causal_masks):
                    causal_mask = stage_causal_masks[stage_idx]
                else:
                    causal_mask = base_causal_mask
            else:
                causal_mask = base_causal_mask

            x = layer(x, causal_mask, key_padding_mask, current_visual, current_visual_pos_masks)

        output = self.ln_f(x)
        logits = self.lm_head(output)
        return logits

    def _create_vision_aware_mask(self, seq_len: int, vision_len: int, device, dtype) -> torch.Tensor:
        """Create vision-aware attention mask

        - Vision part (0:V): Fully bidirectional (can see each other)
        - Text part (V:N):
          - Can see all Vision tokens
          - Use causal mask for other Text tokens
        """
        causal_mask = torch.zeros((seq_len, seq_len), device=device, dtype=dtype)

        # Text-to-Text: Causal mask (can only see previous text)
        text_start = vision_len
        if text_start < seq_len:
            text_causal = torch.triu(
                torch.full((seq_len - text_start, seq_len - text_start), float('-inf'), device=device, dtype=dtype),
                diagonal=1
            )
            causal_mask[text_start:, text_start:] = text_causal

        # Vision-to-Vision: Fully bidirectional (already 0, no change needed)
        # Text-to-Vision: Fully visible (already 0, no change needed)
        # Vision-to-Text: Not visible (maintain causality)
        if text_start < seq_len:
            causal_mask[:text_start, text_start:] = float('-inf')

        return causal_mask


# ==================== Adaptive Pool Embedding (replaces PatchEmbed) ====================
class AdaptivePoolEmbed(nn.Module):
    """
    Use AdaptiveAvgPool3d for feature downsampling to make all stages output the same number of tokens
    Deepest stage does not need this module, can directly flatten
    """
    def __init__(self, output_size: Union[int, Tuple[int, int, int]]):
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size, output_size)
        self.output_size = output_size
        self.pool = nn.AdaptiveAvgPool3d(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, D, H, W] -> [B, N, C]
        # print("before AdaptivePoolEmbed, x.shape=",x.shape)
        x = self.pool(x)
        # print("after AdaptivePoolEmbed, x.shape=",x.shape)
        return x.flatten(2).permute(0, 2, 1)


# ==================== Patch Embedding (retained for backward compatibility) ====================
class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, kernel_size: Union[int, Tuple[int, int, int]], bias: bool = True, padding_mode: str = 'zeros'):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)

        self.kernel_size = kernel_size
        self.padding_mode = padding_mode
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, bias=bias, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 5, f"PatchEmbed expects 5D input [B,C,D,H,W], got {x.shape}"
        B, C, D, H, W = x.shape
        k_d, k_h, k_w = self.kernel_size

        pad_d = (k_d - (D % k_d)) % k_d
        pad_h = (k_h - (H % k_h)) % k_h
        pad_w = (k_w - (W % k_w)) % k_w
        # Use torch.any to avoid TracerWarning
        needs_padding = torch.tensor([pad_d, pad_h, pad_w], device=x.device).any()
        if needs_padding:
            if self.padding_mode == 'zeros':
                x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h, 0, pad_d), mode='constant', value=0)
            else:
                x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h, 0, pad_d), mode=self.padding_mode)

        x = self.proj(x)

        B, C, D_out, H_out, W_out = x.shape
        x = x.flatten(2)
        x = x.permute(0, 2, 1)

        return x


# ==================== LLM Report Generator ====================
class LLMReportGenerator(nn.Module):
    def __init__(self, tokenizer_path: str, llm_embed_dim: int = 512, max_length: int = 8192,
                 num_layers: int = 1, num_heads: int = 8, ffn_dim: int = 2048,
                 dropout: float = 0.0, use_weight_tying: bool = True, use_deepstack: bool = False,
                 use_vision_aware_mask: bool = True,
                 device: Optional[torch.device] = None,
                 vision_bidirectional: bool = True,  # New parameter
                 layers_per_stage: int = 1,          # New: number of layers per stage
                 use_gate: bool = True,              # New: whether to use gating mechanism
                 vision_token_buffer: int = 512):    # New: buffer reserved for vision tokens
        super().__init__()
        from transformers import AutoTokenizer
        import os

        self.llm_embed_dim = llm_embed_dim
        self.max_length = max_length
        self.use_deepstack = use_deepstack
        self.use_vision_aware_mask = use_vision_aware_mask
        self.use_gate = use_gate
        # Don't store device - use property to get it dynamically from parameters

        # Add attributes needed for inference code
        self.llm_model_path = tokenizer_path  # Use tokenizer_path as model_path
        self._model_loaded = False  # Track whether model is loaded

        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, model_max_length=max_length, padding_side="right",
            trust_remote_code=True, use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Special identifiers for vision tokens are needed regardless of use_deepstack
        # This allows the model to recognize variable-length vision token sequences
        special_tokens = ["<|vision_start|>", "<|vision_end|>", "<|vision_pad|>"]
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        self.vision_start_token_id = self.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self.vision_end_token_id = self.tokenizer.convert_tokens_to_ids("<|vision_end|>")

        vocab_size = len(self.tokenizer)
        # Critical fix: RoPE's max_seq_len must include space for vision tokens
        # Actual sequence = visual_tokens + 2(special) + text_tokens
        # text_tokens up to max_length, vision tokens need extra space
        transformer_max_seq_length = max_length + vision_token_buffer
        print(f"[LLMReportGenerator] max_length={max_length}, vision_buffer={vision_token_buffer}, "
              f"transformer_max_seq_length={transformer_max_seq_length}")
        self.llm_model = SimpleTransformerDecoder(
            vocab_size, llm_embed_dim, num_layers, num_heads, ffn_dim,
            transformer_max_seq_length, dropout, use_weight_tying, use_deepstack, use_vision_aware_mask,
            vision_bidirectional,  # Pass new parameter
            layers_per_stage,      # Pass layers_per_stage parameter
            use_gate               # Pass use_gate parameter
        )
        self.eos_token_id = self.tokenizer.eos_token_id or self.tokenizer.pad_token_id

        # Model created during initialization, mark as loaded
        self._model_loaded = True

    @property
    def device(self):
        """Dynamically get device from model parameters."""
        return next(self.parameters()).device

    def forward(self, vision_feature: torch.Tensor, text_prompt: Optional[str] = None,
                text_ids: Optional[torch.Tensor] = None, text_attention_mask: Optional[torch.Tensor] = None,
                generate: bool = False, max_new_tokens: int = 512,
                temperature: float = 0.7, top_p: float = 0.9, top_k: int = 50,
                deepstack_features: Optional[List[torch.Tensor]] = None,
                deepstack_vision_lengths: Optional[List[int]] = None,
                skip_special_tokens: bool = False):

        if generate:
            return self._generate(vision_feature, text_prompt, max_new_tokens,
                                temperature, top_p, top_k, deepstack_features,
                                deepstack_vision_lengths, skip_special_tokens)
        else:
            return self._train_forward(vision_feature, text_ids, text_attention_mask,
                                      deepstack_features, deepstack_vision_lengths)
    
    _debug_printed = False  # Class-level flag, print only once

    def _train_forward(self, vision_feature, text_ids, text_attention_mask,
                       deepstack_features, deepstack_vision_lengths):
        B = vision_feature.shape[0]
        text_embeds = self.llm_model.token_embedding(text_ids)

        # Add special identifiers before and after vision tokens regardless of use_deepstack
        vs_embed = self.llm_model.token_embedding(
            torch.tensor([[self.vision_start_token_id]], device=self.device)
        ).expand(B, 1, -1)
        ve_embed = self.llm_model.token_embedding(
            torch.tensor([[self.vision_end_token_id]], device=self.device)
        ).expand(B, 1, -1)

        # Ensure all features match text_embeds dtype (which comes from embedding layer)
        target_dtype = text_embeds.dtype
        vision_feature = vision_feature.to(dtype=target_dtype, device=self.device)
        if deepstack_features is not None:
            deepstack_features = [f.to(dtype=target_dtype, device=self.device) for f in deepstack_features]

        combined_embeds = torch.cat([vs_embed, vision_feature, ve_embed, text_embeds], dim=1)

        # Debug: print sequence length information (only once)
        if not LLMReportGenerator._debug_printed:
            LLMReportGenerator._debug_printed = True
            total_seq_len = combined_embeds.shape[1]
            visual_tokens = vision_feature.shape[1]
            text_tokens = text_embeds.shape[1]
            max_seq_len = self.max_length
            print(f"[LLM Sequence] visual={visual_tokens} + special=2 + text={text_tokens} = {total_seq_len} (max={max_seq_len})")
            if total_seq_len > max_seq_len:
                print(f"[LLM Sequence] WARNING: exceeds max by {total_seq_len - max_seq_len} tokens!")
        vision_len = 2 + vision_feature.shape[1]
        vision_attn = torch.ones(B, vision_len, device=self.device, dtype=text_attention_mask.dtype)
        combined_attn = torch.cat([vision_attn, text_attention_mask], dim=1)

        visual_pos_masks = torch.zeros(B, combined_embeds.shape[1], dtype=torch.bool, device=self.device)
        visual_pos_masks[:, 1:1+vision_feature.shape[1]] = True

        logits = self.llm_model(inputs_embeds=combined_embeds, attention_mask=combined_attn,
                               visual_pos_masks=visual_pos_masks,
                               deepstack_visual_embeds=deepstack_features,
                               deepstack_vision_lengths=deepstack_vision_lengths)
        text_start_idx = vision_len

        result = {'llm_logits': logits, 'text_start_idx': text_start_idx}
        return result
    
    def _generate(self, vision_feature, text_prompt, max_new_tokens, temperature, top_p, top_k,
                  deepstack_features, deepstack_vision_lengths, skip_special_tokens: bool = False):
        B = vision_feature.shape[0]

        prompt_ids = self.tokenizer(text_prompt, return_tensors='pt', truncation=True,
                                    max_length=self.max_length//2).to(self.device)
        text_embeds = self.llm_model.token_embedding(prompt_ids['input_ids'])

        if text_embeds.shape[0] != B:
            text_embeds = text_embeds.expand(B, -1, -1)

        # Add special identifiers before and after vision tokens regardless of use_deepstack
        # This allows the model to clearly identify vision token boundaries
        vs_embed = self.llm_model.token_embedding(
            torch.tensor([[self.vision_start_token_id]], device=self.device)
        ).expand(B, 1, -1)
        ve_embed = self.llm_model.token_embedding(
            torch.tensor([[self.vision_end_token_id]], device=self.device)
        ).expand(B, 1, -1)
        combined_embeds = torch.cat([vs_embed, vision_feature, ve_embed, text_embeds], dim=1)

        generated_ids = []
        current_embeds = combined_embeds

        # Fixed positions for vision tokens: start at index 1 (skip vision_start), length = vision_feature.shape[1]
        vision_token_length = vision_feature.shape[1]

        for _ in range(max_new_tokens):
            # Set vision position mask regardless of deepstack usage
            visual_pos_masks = torch.zeros(B, current_embeds.shape[1], dtype=torch.bool, device=self.device)
            visual_pos_masks[:, 1:1+vision_token_length] = True

            logits = self.llm_model(inputs_embeds=current_embeds, visual_pos_masks=visual_pos_masks,
                                   deepstack_visual_embeds=deepstack_features,
                                   deepstack_vision_lengths=deepstack_vision_lengths)

            next_logits = logits[:, -1, :] / temperature

            if top_k > 0:
                top_k_vals, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < top_k_vals[:, [-1]]] = float('-inf')

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logits[indices_to_remove] = float('-inf')

            probs = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            generated_ids.append(next_id)

            if (next_id == self.eos_token_id).all():
                break

            next_embed = self.llm_model.token_embedding(next_id)
            current_embeds = torch.cat([current_embeds, next_embed], dim=1)

        generated_ids = torch.cat(generated_ids, dim=1) if generated_ids else torch.empty(B, 0, dtype=torch.long, device=self.device)
        texts = [self.tokenizer.decode(generated_ids[i].cpu().tolist(), skip_special_tokens=skip_special_tokens)
                 for i in range(B)]
        return texts


# ==================== Main Network ====================
class UVLM(nn.Module):
    def __init__(self, in_channels: int, patch_kernel_sizes: List[Union[int, Tuple[int, int, int]]],
                 enable_report_gen: bool = True,
                 tokenizer_path: str = "/path/to/tokenizer/",
                 llm_embed_dim: int = 512, report_max_length: int = 8192,
                 num_heads: int = 8, ffn_dim: int = 2048,
                 dropout: float = 0.0, use_weight_tying: bool = True, use_deepstack: bool = False,
                 use_vision_aware_mask: bool = True,
                 generation_temperature: float = 0.7, generation_top_p: float = 0.9,
                 generation_top_k: int = 50,
                 # Encoder params
                 n_stages: int = 6, features_per_stage: List[int] = [16, 32, 64, 128, 256, 320],
                 kernel_sizes: List[Union[Tuple[int, int, int], int]] = None,
                 strides: List[Union[Tuple[int, int, int], int]] = None,
                 n_blocks_per_stage: List[int] = None,
                 norm_op: nn.Module = nn.InstanceNorm3d,
                 norm_op_kwargs: dict = None,
                 conv_op: nn.Module = nn.Conv3d,
                 conv_bias: bool = True,
                 nonlin: nn.Module = nn.LeakyReLU,
                 nonlin_kwargs: dict = None,
                 dropout_op=None, dropout_op_kwargs=None,
                 use_stages: Optional[List[int]] = None,
                 device: Optional[torch.device] = None,
                 # Classification params
                 only_cls: bool = False,
                 cls_head_num_classes_list: List[int] = [1, 13],
                 cls_drop_out_list: List[float] = [0.0, 0.0],
                 cls_query_num_list: List[int] = [2, 16],
                 use_cross_attention: bool = True,
                 # DeepStack params
                 deepstack_skip_stages: int = 0,
                 layers_per_stage: int = 1,
                 use_gate: bool = False,
                 use_adaptive_pool: bool = True,
                 pool_output_size: Tuple[int, int, int] = None,
                 pool_output_size_list: List[Tuple[int, int, int]] = None,  # Multi-scale vision tokens: pool size for each stage
                 visual_token_length_source_stage: int = -1,  # -1: deepest stage, -2: second deepest, etc.
                 # Mode param: 'only_cls', 'only_report', 'both'
                 mode: str = 'both'):
        super().__init__()

        # Use mode parameter to uniformly control component creation
        self.mode = mode
        self.enable_cls = mode in ['only_cls', 'both']
        self.enable_report_gen = mode in ['only_report', 'both']

        # Keep only_cls for backward compatibility
        self.only_cls = (mode == 'only_cls')
        self.use_deepstack = use_deepstack
        self.use_stages = use_stages or list(range(n_stages))
        self.deepstack_skip_stages = deepstack_skip_stages
        self.layers_per_stage = layers_per_stage
        self.use_gate = use_gate
        self.use_adaptive_pool = use_adaptive_pool
        self.generation_temperature = generation_temperature
        self.generation_top_p = generation_top_p
        self.generation_top_k = generation_top_k
        self.n_stages = n_stages
        self.pool_output_size = pool_output_size
        self.pool_output_size_list = pool_output_size_list
        self.visual_token_length_source_stage = visual_token_length_source_stage
   
        # Automatically calculate num_layers based on deepstack settings
        if use_deepstack:
            # Deepstack mode: num_layers = (total stages - deepstack_skip_stages) * layers_per_stage
            actual_deepstack_stages = n_stages - deepstack_skip_stages
            num_layers = actual_deepstack_stages * layers_per_stage
        else:
            # Non-deepstack mode: only one stage (deepest), so num_layers = layers_per_stage
            num_layers = layers_per_stage

        # Use ResidualEncoder
        # Weight naming: encoder.stem.xxx, encoder.stages.0.xxx, encoder.stages.1.xxx, ...
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

        # Classification heads: only created in enable_cls mode
        self.cls_head_list = nn.ModuleList()
        if self.enable_cls:
            self.cls_drop_out_list = cls_drop_out_list
            self.cls_query_num_list = cls_query_num_list

            for i in range(len(cls_head_num_classes_list)):
                cls_head_num_classes = cls_head_num_classes_list[i]
                cls_drop_out = self.cls_drop_out_list[i]
                cls_query_num = self.cls_query_num_list[i]

                # Use modified ClassificationHead
                self.cls_head_list.append(ClassificationHead(
                    embed_dim=final_features,
                    query_num=cls_query_num,
                    num_classes=cls_head_num_classes,
                    dropout=cls_drop_out,
                    use_cross_attention=use_cross_attention,
                    num_heads=4
                ))

        if self.enable_report_gen:
            # Determine which stages need deepstack (skip first N stages)
            self.deepstack_stages = [stage_idx for stage_idx in self.use_stages
                                   if stage_idx >= self.deepstack_skip_stages]

            # Deepest stage index (no pooling, directly flatten)
            self.deepest_stage_idx = self.deepstack_stages[-1] if self.deepstack_stages else -1

            # Determine source stage index (for multi-scale vision tokens)
            if visual_token_length_source_stage < 0:
                self.source_stage_idx = n_stages + visual_token_length_source_stage
            else:
                self.source_stage_idx = visual_token_length_source_stage
            self.source_stage_idx = max(0, min(self.source_stage_idx, n_stages - 1))

            # Create pool embedding for non-deepest stages (deepest stage directly flattened)
            self.pool_embed_list = nn.ModuleList()
            self.pool_norm_list = nn.ModuleList()
            self.vision_projection_list = nn.ModuleList()
            self.vision_norm_list = nn.ModuleList()

            for i, stage_idx in enumerate(self.deepstack_stages):
                stage_features = features_per_stage[stage_idx]

                # Multi-scale vision tokens mode: use pool_output_size_list
                # Stages deeper than source stage keep original size (None), others pool to source stage size
                if pool_output_size_list is not None:
                    stage_pool_size = pool_output_size_list[stage_idx] if stage_idx < len(pool_output_size_list) else None
                else:
                    # Compatible with old mode: deepest stage not pooled, others pooled to pool_output_size
                    is_deepest = (stage_idx == self.deepest_stage_idx)
                    stage_pool_size = None if is_deepest else pool_output_size

                # Pool embedding: None means no pooling (direct flatten)
                if stage_pool_size is None:
                    self.pool_embed_list.append(None)  # Placeholder, directly flatten
                elif use_adaptive_pool:
                    self.pool_embed_list.append(AdaptivePoolEmbed(stage_pool_size))
                else:
                    self.pool_embed_list.append(
                        PatchEmbed(stage_features, stage_features, patch_kernel_sizes[stage_idx], bias=True)
                    )
                self.pool_norm_list.append(nn.LayerNorm(stage_features))

                # Vision projection for each stage
                vision_proj = nn.Sequential(
                    nn.Linear(stage_features, llm_embed_dim * 2, bias=True),
                    nn.GELU(),
                    nn.Linear(llm_embed_dim * 2, llm_embed_dim, bias=True),
                )
                for module in vision_proj:
                    if isinstance(module, nn.Linear):
                        nn.init.xavier_uniform_(module.weight)
                        nn.init.zeros_(module.bias)
                self.vision_projection_list.append(vision_proj)
                self.vision_norm_list.append(nn.LayerNorm(llm_embed_dim, eps=1e-6))

            # Dynamically compute vision_token_buffer based on pool_output_size (feature map size of source stage)
            # pool_output_size is computed by trainer based on visual_token_length_source_stage
            # For vs2 (vision_stage=-2), pool_output_size is typically (12, 16, 16) = 3072 tokens
            vision_token_buffer = 512  # Default value
            if pool_output_size is not None:
                expected_vision_tokens = pool_output_size[0] * pool_output_size[1] * pool_output_size[2]
                # Buffer must accommodate visual tokens + special tokens (2) + some margin
                vision_token_buffer = max(512, expected_vision_tokens + 128)
                print(f"[ResEncoderUVLM] pool_output_size={pool_output_size}, "
                      f"expected_vision_tokens={expected_vision_tokens}, "
                      f"vision_token_buffer={vision_token_buffer}")

            self.llm_report_gen = LLMReportGenerator(
                tokenizer_path, llm_embed_dim, report_max_length, num_layers, num_heads,
                ffn_dim, dropout, use_weight_tying, use_deepstack, use_vision_aware_mask, device,
                vision_bidirectional=True, layers_per_stage=layers_per_stage, use_gate=use_gate,
                vision_token_buffer=vision_token_buffer
            )

    @property
    def device(self):
        """Dynamically get device from model parameters."""
        return next(self.parameters()).device

    def forward(self, input_image, generate_report=False, report_prompt=None,
                report_text_ids=None, report_text_attention_mask=None,
                max_new_tokens: int = 512, skip_special_tokens: bool = False,
                temperature: Optional[float] = None, top_p: Optional[float] = None,
                top_k: Optional[int] = None):

        # Use ResidualEncoder to get features from all stages
        all_stage_features = self.encoder(input_image, return_all_stages=True)
        x = all_stage_features[-1]  # Output from the last stage

        # Collect stage features needed for deepstack
        stage_features = []
        if self.enable_report_gen and self.use_deepstack:
            for stage_idx in self.deepstack_stages:
                if stage_idx < len(all_stage_features):
                    stage_features.append(all_stage_features[stage_idx].clone())

        # cls: compute classification logits only in enable_cls mode
        cls_pred_list = []
        if self.enable_cls:
            for cls_head in self.cls_head_list:
                cls_pred_list.append(cls_head(x))

        if not self.enable_report_gen:
            return None, cls_pred_list

        # Generate deepstack features
        deepstack_features = None
        deepstack_vision_lengths = None  # Multi-scale vision tokens: token count per stage
        if self.use_deepstack:
            deepstack_features = []
            deepstack_vision_lengths = []
            for i, stage_idx in enumerate(self.deepstack_stages):
                stage_feat = stage_features[i]
                stage_pool_embed = self.pool_embed_list[i]
                stage_pool_norm = self.pool_norm_list[i]
                stage_vision_proj = self.vision_projection_list[i]
                stage_vision_norm = self.vision_norm_list[i]

                # If pool_embed is None, flatten directly; otherwise use pool_embed
                if stage_pool_embed is None:
                    # [B, C, D, H, W] -> [B, N, C]
                    vision_tokens = stage_feat.flatten(2).permute(0, 2, 1)
                else:
                    vision_tokens = stage_pool_embed(stage_feat)
                vision_tokens = stage_pool_norm(vision_tokens)
                vision_tokens = stage_vision_proj(vision_tokens)
                vision_tokens = stage_vision_norm(vision_tokens)
                deepstack_features.append(vision_tokens)
                deepstack_vision_lengths.append(vision_tokens.shape[1])

            # U-Net Style: deep encoder → shallow decoder, shallow encoder → deep decoder
            deepstack_features = deepstack_features[::-1]
            deepstack_vision_lengths = deepstack_vision_lengths[::-1]
        else:
            # Non-deepstack mode: directly use visual features from source stage (no pooling)
            stage_feat = all_stage_features[self.source_stage_idx]
            vision_tokens = stage_feat.flatten(2).permute(0, 2, 1)  # [B, C, D, H, W] -> [B, N, C]

            # Find the corresponding projection layer
            source_idx_in_list = 0
            for i, stage_idx in enumerate(self.deepstack_stages):
                if stage_idx == self.source_stage_idx:
                    source_idx_in_list = i
                    break

            vision_tokens = self.pool_norm_list[source_idx_in_list](vision_tokens)
            vision_tokens = self.vision_projection_list[source_idx_in_list](vision_tokens)
            vision_tokens = self.vision_norm_list[source_idx_in_list](vision_tokens)

        if self.use_deepstack:
            # Deepstack mode uses features from source_stage as primary visual input
            # Reversed index calculation:
            #   Original order: [stage0, stage1, ..., stageN-1] (shallow→deep)
            #   Reversed order: [stageN-1, stageN-2, ..., stage0] (deep→shallow)
            # Position of source_stage_idx in reversed order = (N-1) - source_stage_idx
            # where N = len(deepstack_stages)
            num_deepstack_stages = len(self.deepstack_stages)
            # Find position of source_stage_idx in deepstack_stages
            source_idx_in_deepstack = -1
            for i, stage_idx in enumerate(self.deepstack_stages):
                if stage_idx == self.source_stage_idx:
                    source_idx_in_deepstack = i
                    break

            if source_idx_in_deepstack >= 0:
                # Reversed index
                reversed_idx = num_deepstack_stages - 1 - source_idx_in_deepstack
                main_vision_tokens = deepstack_features[reversed_idx]
            else:
                # fallback: use deepest layer (first in reversed order)
                main_vision_tokens = deepstack_features[0]

            # Padding scheme: pad all deepstack_features to same length as main_vision_tokens
            # This ensures uniform token count across all stages, simplifying subsequent processing
            main_vision_len = main_vision_tokens.shape[1]
            padded_features = []
            for f in deepstack_features:
                if f.shape[1] < main_vision_len:
                    # Pad with zeros to target length
                    padding = torch.zeros(
                        f.shape[0], main_vision_len - f.shape[1], f.shape[2],
                        dtype=f.dtype, device=f.device
                    )
                    f = torch.cat([f, padding], dim=1)
                padded_features.append(f)
            deepstack_features = padded_features
            # Update vision_lengths to uniform length
            deepstack_vision_lengths = [main_vision_len] * len(deepstack_features)
        else:
            main_vision_tokens = vision_tokens

        if generate_report:
            # Use provided parameters; if not provided, use default instance attributes
            use_temperature = temperature if temperature is not None else self.generation_temperature
            use_top_p = top_p if top_p is not None else self.generation_top_p
            use_top_k = top_k if top_k is not None else self.generation_top_k

            report_output = self.llm_report_gen(
                main_vision_tokens, report_prompt, generate=True,
                max_new_tokens=max_new_tokens,
                temperature=use_temperature,
                top_p=use_top_p,
                top_k=use_top_k,
                deepstack_features=deepstack_features,
                deepstack_vision_lengths=deepstack_vision_lengths,
                skip_special_tokens=skip_special_tokens
            )
        else:
            report_output = self.llm_report_gen(
                main_vision_tokens, text_ids=report_text_ids,
                text_attention_mask=report_text_attention_mask,
                generate=False,
                deepstack_features=deepstack_features,
                deepstack_vision_lengths=deepstack_vision_lengths
            )

        return report_output, cls_pred_list
