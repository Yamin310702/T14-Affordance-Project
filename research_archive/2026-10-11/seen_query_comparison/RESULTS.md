# OpenAD seen-name vs held-out-query comparison

Completed 2026-10-11. This fills the earlier audit gap: the released OpenAD checkpoint had only been evaluated with its held-out validation query names in E01/E05.

## What was compared

I ran the released OpenAD PN2 partial-view checkpoint against the official AffordanceNet validation data. For each point cloud, the OpenAD point features were computed once and scored against two 19-name text vocabularies. The aligned class slots keep the ground-truth label indices fixed; only the query names change. The conditions are:

- **Training-name queries:** `grasp`, `contain`, `lift`, `openable`, `layable`, `sittable`, `support`, `wrap_grasp`, `pourable`, `move`, `displaY`, `pushable`, `pull`, `listen`, `wear`, `press`, `cut`, `stab`, `none`.
- **Held-out query names:** `grab`, `accommodate`, `raise`, `unlock`, `rest`, `take a seat`, `bear`, `wrap`, `pour`, `reposition`, `demonstrate`, `push`, `drag`, `hear`, `clothe`, `thumb`, `slice`, `jab`, `none`.

The same geometry features were used for each pair. The paired implementation was checked against the original OpenAD forward pass: predicted point labels matched exactly on a check object for both vocabularies. The checkpoint SHA-256 is `cbcfe9ff70448d97a7047f1ade817645a9be29dd2782fba57776792a8210f2ae`.

## Main results

IoU is computed globally from all points in a split, matching the official evaluator's point-wise class IoU calculation. The 19-class mean includes `none`; the 18-affordance mean excludes it. Values are percentages.

| Input setting | Validation inputs | Training-name mIoU (19 / 18) | Held-out-name mIoU (19 / 18) | Held-out minus training (19 / 18) | Point accuracy (training / held-out) |
|---|---:|---:|---:|---:|---:|
| Partial views | 9,140 views / 2,285 shapes | 41.45 / 40.39 | 12.53 / 10.76 | −28.92 / −29.62 points | 69.75 / 45.24 |
| Full shapes | 2,285 shapes | 36.18 / 35.18 | 9.75 / 7.87 | −26.43 / −27.31 points | 64.08 / 44.33 |

On partial views, the held-out-name 19-class mIoU of 12.531% closely reproduces the previous E05 result of 12.536% (which used batch size 1 on MPS). The new paired run used batch size 4 on CPU, and shares one backbone pass within each query pair.

The class-level results are in `partial_validation/per_class_iou.csv` and `full_validation/per_class_iou.csv`. The aligned name pairs generally lose IoU when the held-out query name replaces the training name. In the partial setting, examples include `grasp` 44.43% vs `grab` 1.02%, `openable` 27.91% vs `unlock` 4.09%, and `cut` 58.65% vs `slice` 53.89%. Some pairs are closer: `listen` 52.62% vs `hear` 51.84%. Do not describe every class as worse; inspect the saved table for the full per-class values.

## Interpretation and limits

The result demonstrates strong **query-vocabulary sensitivity** in this released checkpoint: using the names associated with training yields much higher localization scores than substituting the config's held-out validation names. It does not demonstrate that training-name querying is a novel method, or establish generalization to truly unseen affordance concepts. Many aligned pairs are near-synonyms, and `none` is the same query in both conditions. Report both the 19-class metric and the 18-action metric so the shared `none` class does not inflate or obscure the affordance result.

This is a validation-set diagnostic, not a held-out test-set estimate. Partial validation contains four views per shape, so those 9,140 views are not 9,140 independent objects. As a secondary paired analysis, averaging the saved per-view macro IoU within each of the 2,285 shapes and bootstrapping shapes (10,000 resamples, seed 20261011) gave held-out-minus-training 18-class mean per-view IoU differences of −4.11 percentage points (95% interval −4.23 to −3.99) for partial views and −4.41 points (−4.51 to −4.32) for full shapes. This equal-shape secondary estimand differs from the point-pooled primary mIoU above.

The run used CPU because MPS was not available to this Python process. No model weights were trained or changed. The full-shape evaluation uses the same partial-view-trained checkpoint and the same validation shapes; it is a full-input control, not an independent dataset.

The complete MPS rerun reproduced these headline metrics within at most 0.0054 percentage points and took less time on this Mac. See the [CPU/MPS comparison](../seen_query_comparison_mps/DEVICE_COMPARISON.md) and its saved MPS per-class/per-view outputs.

## Saved files

- `partial_validation/run.json`, `per_class_iou.csv`, `per_view_metrics.csv`
- `full_validation/run.json`, `per_class_iou.csv`, `per_view_metrics.csv`
- `pilot/run_seen_query_comparison.py` — reproducible runner; expects the OpenAD repository and checkpoint at the paths recorded in `run.json`.
- `seen_query_partial_run.log`, `seen_query_full_run.log` — console logs.
