# Released Checkpoints

One final-epoch checkpoint is provided for each dataset:

https://huggingface.co/yinghemedical/U-VLM

The checkpoints are retrained models. Metrics below are recomputed from those
files and differ from the paper table.

Code is Apache-2.0. Released weights are CC BY-NC 4.0 (non-commercial use
only).

## [CT-RATE](https://huggingface.co/datasets/ibrahimhamamci/CT-RATE)

- Target: narrative report (`report` / `report_narrative`)
- Language decoder: 0.1B, trained from scratch
- Checkpoint: `checkpoint_final.pth`
- Evaluation split: CT-RATE `valid_fixed` (3,039 cases)
- Generation: `max_new_tokens=512`

| Metric | Value |
|---|---:|
| Macro F1 (RadBERT) | 0.4506 |
| BLEU-1 | 0.4818 |
| BLEU-2 | 0.3595 |
| BLEU-3 | 0.2845 |
| BLEU-4 | 0.2358 |
| BLEU-mean | 0.3404 |

Paper reference: F1 0.414, BLEU-mean 0.349.

## [AbdomenAtlas 3.0](https://huggingface.co/datasets/AbdomenAtlas/AbdomenAtlas3.0Mini)

- Target: structured report (`report_structured`)
- Language decoder: 0.1B, trained from scratch
- Checkpoint: `checkpoint_final.pth`
- Evaluation split: AbdomenAtlas 3.0 test split used for the paper (926 cases)
- Generation: `max_new_tokens=512`

| Metric | Value |
|---|---:|
| Macro F1 (DeepSeek label extraction) | 0.5791 |
| BLEU-1 | 0.5562 |
| BLEU-2 | 0.4629 |
| BLEU-3 | 0.3916 |
| BLEU-4 | 0.3240 |
| BLEU-mean | 0.4337 |

Paper reference: F1 0.624, BLEU-mean 0.437 (structured main row).

## Inference Outputs

The weight repository includes full four-column prediction CSVs:

- [`inference_results/ct_rate_validation.csv`](https://huggingface.co/yinghemedical/U-VLM/blob/main/inference_results/ct_rate_validation.csv) — 3,039 cases.
- [`inference_results/abdomenatlas3_test.csv`](https://huggingface.co/yinghemedical/U-VLM/blob/main/inference_results/abdomenatlas3_test.csv) — 926 cases.

Columns: `case`, `resized_image_shape`, `ground_truth_report`,
`predicted_report`. Predictions use `max_new_tokens=512`.

## Evaluation Protocol

- CT-RATE F1: RadBERT report-label extraction, 18 classes, threshold 0.5.
- AbdomenAtlas F1: DeepSeek-based 3-class lesion extraction, following the
  paper protocol.
- BLEU: `pycocoevalcap`.
- CT-RATE reports are scored against the official report text; AbdomenAtlas
  structured reports are scored against `report_structured`.

The released CT-RATE model is a narrative-report model. The released
AbdomenAtlas model is a structured-report model.

No CT-RATE or AbdomenAtlas images, reports, or patient identifiers are included
in the weight repository.
