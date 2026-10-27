# E10: held-out pour-query sensitivity

Status: complete. Objects: 20, unique shapes. Model: Unofficial Espresso-3D; Affogato-pretrained weights.

## Equal-weight mean across selected development objects

| Condition | aIoU % | AUC % | SIM % | MAE |
|---|---:|---:|---:|---:|
| original | 9.47 | 74.98 | 36.40 | 0.3443 |
| phrase_1 | 9.99 | 74.94 | 36.59 | 0.3506 |
| phrase_2 | 8.90 | 74.64 | 36.20 | 0.3346 |
| phrase_3 | 6.18 | 55.63 | 31.28 | 0.3677 |
| ensemble | 8.01 | 71.85 | 35.49 | 0.3510 |

## Paired aIoU difference from original query

| Condition | Mean change, percentage points | Paired bootstrap 95% interval |
|---|---:|---:|
| phrase_1 | +0.52 | [-0.31, +1.35] |
| phrase_2 | -0.57 | [-1.50, +0.72] |
| phrase_3 | -3.29 | [-5.89, -1.32] |
| ensemble | -1.46 | [-2.29, -0.78] |

The objects were selected deterministically from those with a positive ground-truth pour region; this is a development subset, not a random test set. The interval is descriptive and does not make this selected sample representative. The model is an unofficial Espresso-3D reimplementation. Affordance-query thresholds use the common repository metric; OpenAD multiclass mIoU is not directly comparable. The fixed 0.5 diagnostic can produce empty positive masks; read recall and false-positive rates together.

See run.json for prompts, object IDs, per-object metric values and exact selection rule; predictions.npz contains the pointwise heatmaps and GT.
