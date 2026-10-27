# E12 CPU vs MPS rerun comparison

**The MPS rerun reproduces the CPU experiment.** Both runs evaluated the same 100 objects (20 each for grab, pour, wrap, accommodate, and unlock), the same 500 object-condition rows, prompts, annotations, and pretrained weights. XYZ and ground-truth arrays match exactly.

## Result

Mean aIoU, AUC, SIM, and MAE are the same to the displayed precision for every affordance and query condition. Across the 500 individual result rows, maximum absolute differences were:

| Metric | Maximum absolute difference |
|---|---:|
| aiou | 7.0955408e-06 |
| auc | 2.0586298e-05 |
| sim | 2.1457672e-06 |
| mae | 8.9406967e-07 |

Across prediction heatmaps, mean absolute CPU–MPS difference was 2.37e-07, 95th percentile 7.45e-07, maximum 4.05e-06. These tiny floating-point differences do not change the study’s conclusions.

The measured inference loop took 330.5 seconds on CPU and 89.3 seconds on MPS (3.70× faster on MPS). This excludes data selection and model startup.

## Mean aIoU comparison

| Affordance | Condition | CPU | MPS | MPS − CPU (percentage points) |
|---|---|---:|---:|---:|
| grab | original | 11.1306% | 11.1306% | +0.00000 |
| grab | phrase_1 | 11.2439% | 11.2439% | -0.00004 |
| grab | phrase_2 | 10.9587% | 10.9587% | +0.00000 |
| grab | phrase_3 | 9.7985% | 9.7985% | +0.00000 |
| grab | ensemble | 10.3880% | 10.3880% | +0.00000 |
| pour | original | 11.9206% | 11.9206% | +0.00000 |
| pour | phrase_1 | 12.8352% | 12.8352% | +0.00000 |
| pour | phrase_2 | 12.7267% | 12.7267% | +0.00000 |
| pour | phrase_3 | 6.5247% | 6.5247% | +0.00000 |
| pour | ensemble | 10.3321% | 10.3321% | +0.00000 |
| wrap | original | 6.8330% | 6.8330% | +0.00000 |
| wrap | phrase_1 | 6.9427% | 6.9427% | +0.00000 |
| wrap | phrase_2 | 5.1481% | 5.1481% | +0.00000 |
| wrap | phrase_3 | 7.2732% | 7.2732% | +0.00000 |
| wrap | ensemble | 6.3412% | 6.3412% | +0.00000 |
| accommodate | original | 2.0907% | 2.0907% | +0.00000 |
| accommodate | phrase_1 | 3.6131% | 3.6131% | +0.00000 |
| accommodate | phrase_2 | 3.4651% | 3.4651% | +0.00000 |
| accommodate | phrase_3 | 3.7754% | 3.7754% | +0.00000 |
| accommodate | ensemble | 3.2176% | 3.2176% | +0.00000 |
| unlock | original | 3.7591% | 3.7591% | +0.00000 |
| unlock | phrase_1 | 4.6563% | 4.6563% | +0.00000 |
| unlock | phrase_2 | 4.2838% | 4.2838% | +0.00000 |
| unlock | phrase_3 | 4.6989% | 4.6989% | +0.00000 |
| unlock | ensemble | 4.4899% | 4.4899% | +0.00000 |

MPS was available when the run was launched directly outside the restricted Codex shell. The MPS results and heatmaps are saved separately, leaving the original CPU run unchanged. See `device_comparison.json` for precision-level comparisons.
