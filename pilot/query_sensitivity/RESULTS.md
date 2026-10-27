# E08: pouring-query sensitivity on a partial-view vase

Run on 2026-10-08 using the adapted unofficial Espresso-3D Affogato-pretrained model on MPS. Input: saved OpenAD partial validation sample 452, Vase, 2,048 points, 563 points annotated pour. This is a previously inspected development example, not a random sample or final benchmark.

## Fixed experiment

Same weights, geometry, input point order, normalization and cached point features. The original query was rerun as a control. Three predeclared phrasings were tested, then their sigmoid heatmaps were averaged. No prompts or thresholds were revised after viewing the results. The original query is not included in this three-query average.

| Condition | Query | aIoU (%) | AUC (%) | MAE (lower better) |
|---|---|---:|---:|---:|
| Original | Point to the part you would use to pour. | 16.42 | 91.02 | 0.2837 |
| Query 1 | Point to the part used for pouring. | 18.20 | 91.19 | 0.2837 |
| Query 2 | Locate the region from which liquid is poured. | 13.57 | 89.75 | 0.2826 |
| Query 3 | Identify the pouring region. | 7.79 | 56.18 | 0.3507 |
| Ensemble | Mean of Query 1, Query 2 and Query 3 scores | 12.45 | 91.69 | 0.3056 |

Metrics use the repository's sample_metrics implementation with binary pour labels. aIoU averages point IoU over 20 equally spaced thresholds from 0 to 1. This differs from the original OpenAD multiclass mIoU, so those numbers must not be ranked together.

## Interpretation

Changing wording altered predictions on this input. Query 1 improved aIoU by 1.78 percentage points versus the original; Query 3 was much worse. The ensemble reduced aIoU by 3.98 percentage points and increased MAE. Thus this case does not support the proposed localization improvement from simple averaging.

Ensemble AUC was slightly higher, indicating somewhat better point ranking, while absolute score-dependent localization was worse. Do not summarize that as an overall improvement. At the fixed diagnostic threshold 0.5, the ensemble predicted zero positive points. The primary metric was fixed threshold-averaged aIoU, not a threshold selected to favor a method.

Low binary disagreement can be misleading: Query 2 and Query 3 both predicted zero positive points at 0.5, despite differing continuous heatmaps. Their mean absolute score difference was about 0.130 on the 0–1 scale. Therefore identical empty masks do not establish useful robustness.

The original query reproduced the earlier heatmap closely (mean absolute difference approximately 5.29e-8), supporting consistency of this local rerun. This is not a CPU/CUDA numerical-equivalence test.

## Limits and next decision

Single selected object, one affordance, unofficial implementation, adapted precision/device, and cross-dataset input. Intended equivalence of the three phrasings is a human design choice; it has not been independently reviewed. Results motivate testing wording sensitivity on a fixed multi-object development set including successes and failures. Keep final evaluation objects separate. Do not choose only Query 1 for a claimed final gain using this same object.

Provisional proposal framing: investigate how query wording affects localization, and whether any ensemble helps on held-out objects. Do not promise that ensembling improves results. OpenAD's none-confusion diagnosis is separate, and Affostruction remains untested.

Files: predictions.npz (all scores, geometry, GT), run.json (queries, provenance, full metrics), comparison.png (common viewpoint, fixed 0–1 color scale).
