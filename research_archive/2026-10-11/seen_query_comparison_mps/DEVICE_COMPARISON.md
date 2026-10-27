# E17 — CPU/MPS reproducibility comparison

**Status:** both full MPS splits completed on 2026-10-11 using the same released OpenAD PN2 checkpoint, validation data, seed, query lists, and batch size (4) as the archived CPU runs. The MPS runs evaluated 9,140 partial views and 2,285 full shapes. Raw MPS outputs and per-class/per-view CSV files are saved in this folder.

## Result

The two devices produce the same research conclusion: the training-name vocabulary scores far above the aligned held-out query names in both partial and full input settings. Across the headline global mIoU values, CPU-to-MPS differences are at most **0.0054 percentage points**; point-accuracy differences are at most **0.0061 points**. These small changes do not alter the comparison.

| Split | Condition | CPU mIoU (19 / 18) | MPS mIoU (19 / 18) | CPU accuracy | MPS accuracy |
|---|---|---:|---:|---:|---:|
| Partial | Training-name queries | 41.4538 / 40.3862 | 41.4547 / 40.3877 | 69.7535 | 69.7546 |
| Partial | Held-out query names | 12.5310 / 10.7630 | 12.5329 / 10.7653 | 45.2447 | 45.2398 |
| Full | Training-name queries | 36.1835 / 35.1826 | 36.1800 / 35.1794 | 64.0763 | 64.0759 |
| Full | Held-out query names | 9.7538 / 7.8678 | 9.7557 / 7.8699 | 44.3318 | 44.3283 |

The first mIoU column includes all 19 classes, including `none`; the second excludes `none` and averages the 18 affordances. Percentages are shown. The held-out-minus-training mIoU gap also reproduces: approximately −28.9 / −29.6 points on partial inputs and −26.4 / −27.3 points on full inputs (19-class / 18-affordance means).

## Runtime and interpretation

| Split | CPU elapsed | MPS elapsed |
|---|---:|---:|
| Partial | 1,103.6 s | 639.3 s |
| Full | 278.4 s | 159.8 s |

For these runs, MPS took about 42–43% less elapsed time. This is a single-machine timing comparison, not a general hardware benchmark. The tiny metric differences are consistent with floating-point/backend variation. No model weights were trained or changed.

This is a device-reproduction check of the OpenAD validation experiment. It is not a new method result and does not remove the original limitations: these are validation data, many query pairs are semantic aliases, and the comparison establishes sensitivity to the checkpoint's training-associated vocabulary rather than generalization to novel affordance concepts.

## Saved files

- `partial_validation/run.json`, `per_class_iou.csv`, `per_view_metrics.csv`
- `full_validation/run.json`, `per_class_iou.csv`, `per_view_metrics.csv`
- CPU comparison: [seen-query results](../seen_query_comparison/RESULTS.md)
- Runner: `pilot/run_seen_query_comparison.py`
