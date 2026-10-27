# E11: grab-query sensitivity on 20 new objects

**Status:** complete. **Model:** unofficial Espresso-3D reimplementation with Affogato-pretrained weights. **Input:** OpenAD validation partial views. **Objects:** 20 unique shapes with positive ground-truth grab regions, disjoint from E09 and E10.

## Results

Scores are equal-weight means across the 20 objects. aIoU, AUC and SIM are percentages; lower MAE is better.

| Condition | Query / operation | aIoU | AUC | SIM | MAE |
|---|---|---:|---:|---:|---:|
| original | Original: “Point to the part you would grab.” | 12.89% | 65.05% | 43.30% | 0.4072 |
| phrase_1 | “Point to the part you would grasp.” | 16.31% | 64.38% | 47.75% | 0.3813 |
| phrase_2 | “Identify the region suitable for gripping.” | 11.75% | 65.74% | 43.88% | 0.3978 |
| phrase_3 | “Locate where you would grip the object.” | 15.78% | 61.16% | 46.58% | 0.3812 |
| ensemble | Pointwise average of the three paraphrase heatmaps | 14.25% | 67.06% | 46.24% | 0.3868 |

The best mean aIoU was phrase 1 (“grasp”), 16.31%, compared with 12.89% for the original “grab” wording. Phrase 3 reached 15.78%; phrase 2 scored 11.75%. The three-phrase average scored 14.25%, below the best single paraphrase.

## What this supports

The choice of wording appears to affect this model’s localization on this selected grab set. However, the paired 95% bootstrap interval for phrase 1’s +3.43 percentage-point change spans −1.67 to +10.06 points; the interval for phrase 3’s +2.90 points spans −3.12 to +10.52. The small selected pilot therefore does not establish a reliable improvement. The result also differs from pouring, where paraphrases and averaging did not improve reliably. This points toward testing affordance-specific prompt sensitivity, rather than claiming one universal best prompt or that averaging prompts is better.

## Limits

Objects were selected deterministically from samples with a positive ground-truth grab label; they are not a random test sample. Several object categories repeat. The pretrained model comes from Affogato/Espresso and is being evaluated on OpenAD partial point clouds, so this is a cross-dataset exploratory test. Queries are fixed text paraphrases, not human-validated descriptions. The bootstrap intervals describe only these selected objects. Affostruction has not been run. The fixed 0.5 threshold predicts very few positive points for some queries, so thresholded IoU/recall should be interpreted carefully. OpenAD multiclass mIoU is not directly comparable to these Espresso affordance metrics.

## Saved artifacts

- `run.json`: prompts, object IDs, metrics, paired bootstrap intervals, and selection rule.
- `per_object_heatmaps.npz`: XYZ, ground truth, and all five score maps for each object.
- `mean_metrics.csv` and `per_object_metrics.csv`: tabular results.
- `per_object_aiou.png`: per-object comparison.
- `run_grab_query_sensitivity.py` and `summarize_grab_query_sensitivity.py`: experiment scripts.
- `grab_query_sensitivity.log`: execution log.
