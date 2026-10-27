# E13: held-out confirmation of preselected query paraphrases

**Status:** complete on MPS. **Model:** unofficial Espresso-3D with Affogato-pretrained weights. **Data:** OpenAD validation partial views. **Design:** compare the original prompt to one paraphrase selected from E12 before examining E13 results. Each affordance has 40 fresh objects; 200 unique shapes total, disjoint from E09–E12 and from each other within this run. The query pair was fixed in advance, but selection was still limited to objects with a positive ground-truth label.

## Results

aIoU, AUC and SIM are percentages; MAE is lower-is-better. AUC is undefined for one wrap object whose every point is labeled positive, so wrap AUC uses 39 objects; other metrics use 40.

| Affordance | Original aIoU | Selected paraphrase aIoU | Change (pp) | Paired bootstrap 95% interval (pp) | Objects with improved aIoU |
|---|---:|---:|---:|---:|---:|
| grab | 10.61% | 16.46% | +5.85 | [+1.59, +11.00] | 20/40 |
| pour | 11.53% | 11.96% | +0.44 | [-0.12, +0.99] | 27/40 |
| wrap | 11.36% | 10.63% | -0.73 | [-1.35, -0.19] | 18/40 |
| accommodate | 1.95% | 2.29% | +0.34 | [-0.19, +0.93] | 18/40 |
| unlock | 2.37% | 2.93% | +0.56 | [-0.30, +1.65] | 17/40 |

## Locked prompt pairs

| Affordance | Original query | E12-selected paraphrase |
|---|---|---|
| grab | “Point to the part you would grab.” | “Point to the part you would grasp.” |
| pour | “Point to the part you would use to pour.” | “Point to the part used for pouring.” |
| wrap | “Point to the part you can wrap your hand around.” | “Point to the region designed to be held by wrapping your hand around it.” |
| accommodate | “Point to the part of the object that can hold something.” | “Point to the part where you would place or keep something.” |
| unlock | “Point to the part you would use to unlock this object.” | “Point to the part you would operate to gain access.” |

## Interpretation

The grab paraphrase improved mean aIoU by **5.85 percentage points** on this fresh set (paired interval +1.59 to +11.00). The wrap paraphrase reduced it by **0.73 points** (−1.35 to −0.19). Pour (+0.44), accommodate (+0.34), and unlock (+0.56) had intervals that include zero. This supports a tentative, affordance-specific wording effect: a paraphrase helped grab, hurt wrap, and did not produce a clear change for the other three labels.

The intervals are unadjusted across five comparisons, and the evaluation set is a deterministic GT-positive subset from one dataset. Treat this as stronger preliminary evidence than E12, not a final universal claim. The model is unofficial Espresso-3D with Affogato-pretrained weights evaluated cross-dataset on OpenAD; this does not test Affostruction.

## Artifacts

- `run.json`: fixed prompts, selected object IDs, metrics, bootstrap intervals, and run metadata.
- `predictions.npz`: point coordinates, ground truth, and both predicted heatmaps.
- `mean_metrics.csv` and `per_object_metrics.csv`: tabular results.
- `paired_aiou_deltas.png`: paired effect plot.
- `run.log`: execution log.
- `run_e13_confirmation.py` and `summarize_e13_confirmation.py`: scripts.
