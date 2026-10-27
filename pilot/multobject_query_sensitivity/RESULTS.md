# E09: multi-object pour-query sensitivity

Status: complete. Objects: 10, unique shapes. Model: Unofficial Espresso-3D; Affogato-pretrained weights.

## Equal-weight mean across selected development objects

| Condition | aIoU % | AUC % | SIM % | MAE |
|---|---:|---:|---:|---:|
| original | 6.10 | 67.69 | 24.67 | 0.2730 |
| phrase_1 | 6.84 | 69.10 | 24.74 | 0.2856 |
| phrase_2 | 5.25 | 67.16 | 24.12 | 0.2579 |
| phrase_3 | 4.80 | 58.77 | 22.70 | 0.2863 |
| ensemble | 5.46 | 66.29 | 23.90 | 0.2766 |

## Paired aIoU difference from original query

| Condition | Mean change, percentage points | Paired bootstrap 95% interval |
|---|---:|---:|
| phrase_1 | +0.74 | [+0.36, +1.20] |
| phrase_2 | -0.85 | [-1.68, +0.08] |
| phrase_3 | -1.31 | [-2.82, +0.08] |
| ensemble | -0.65 | [-1.22, -0.15] |

The objects were selected deterministically from those with a positive ground-truth pour region; this is a development subset, not a random test set. The interval is descriptive and does not make this selected sample representative. The model is an unofficial Espresso-3D reimplementation. Affordance-query thresholds use the common repository metric; OpenAD multiclass mIoU is not directly comparable. The fixed 0.5 diagnostic can produce empty positive masks; read recall and false-positive rates together.

See run.json for prompts, object IDs, per-object metric values and exact selection rule; predictions.npz contains the pointwise heatmaps and GT.
