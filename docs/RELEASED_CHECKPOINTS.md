# Released Checkpoints

We release one final-epoch checkpoint for each dataset on the Hugging Face
repository:

https://huggingface.co/yinghemedical/U-VLM

These checkpoints were retrained after the paper. The numbers below are
recomputed from the released checkpoints and therefore differ slightly from the
paper table.

## CT-RATE

- Target: narrative report (`report` / `report_narrative`)
- Model: U-VLM 0.1B decoder
- Checkpoint: `checkpoint_final.pth`
- Evaluation split: CT-RATE `valid_fixed`

| Metric | Value |
|---|---:|
| Macro F1 (RadBERT) | 0.4497 |
| BLEU-1 | 0.4819 |
| BLEU-2 | 0.3595 |
| BLEU-3 | 0.2845 |
| BLEU-4 | 0.2359 |
| BLEU-mean | 0.3404 |
| ROUGE-L | 0.3335 |
| CIDEr | 0.1541 |
| CRG | 0.3239 |

Paper reference: F1 0.414, BLEU-mean 0.349.

## AbdomenAtlas 3.0

- Target: structured report (`report_structured`)
- Model: U-VLM 0.1B decoder
- Checkpoint: `checkpoint_final.pth`
- Evaluation split: AbdomenAtlas 3.0 test split used for the paper

| Metric | Value |
|---|---:|
| Macro F1 (DeepSeek label extraction) | 0.5718 |
| BLEU-1 | 0.5569 |
| BLEU-2 | 0.4635 |
| BLEU-3 | 0.3921 |
| BLEU-4 | 0.3244 |
| BLEU-mean | 0.4342 |
| ROUGE-L | 0.6250 |
| CIDEr | 0.1049 |

Paper reference: F1 0.624, BLEU-mean 0.437 (structured main row).

## Evaluation Protocol

- CT-RATE F1: RadBERT report-label extraction, 18 classes, threshold 0.5.
- AbdomenAtlas F1: DeepSeek-based 3-class lesion extraction, following the
  paper protocol.
- BLEU/ROUGE-L/CIDEr: `pycocoevalcap`.
- CT-RATE reports are scored against the official report text; AbdomenAtlas
  structured reports are scored against `report_structured`.

The released CT-RATE model is a narrative-report model, while the released
AbdomenAtlas model is a structured-report model. Their metrics are not directly
comparable across datasets.

No CT-RATE or AbdomenAtlas images, reports, or patient identifiers are included
in the weight repository.
