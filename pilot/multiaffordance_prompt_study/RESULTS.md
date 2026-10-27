# E12: prompt wording across five affordances

**Status:** complete. **Model:** unofficial Espresso-3D with Affogato-pretrained weights. **Device:** CPU (MPS was unavailable during this run). **Data:** OpenAD validation partial views. **Selection:** 20 distinct positive-label objects for each affordance; all objects from E09, E10, and E11 were excluded. The five sets contain 94 unique shape IDs overall; because selection is independent by affordance, some shapes can appear in more than one affordance set.

## Mean results

Each affordance uses 20 objects. aIoU, AUC and SIM are shown as percentages; MAE is lower-is-better.

| Affordance | Original | Phrase 1 | Phrase 2 | Phrase 3 | Paraphrase average |
|---|---:|---:|---:|---:|---:|
| grab | 11.13% | 11.24% | 10.96% | 9.80% | 10.39% |
| pour | 11.92% | 12.84% | 12.73% | 6.52% | 10.33% |
| wrap | 6.83% | 6.94% | 5.15% | 7.27% | 6.34% |
| accommodate | 2.09% | 3.61% | 3.47% | 3.78% | 3.22% |
| unlock | 3.76% | 4.66% | 4.28% | 4.70% | 4.49% |

## Fixed query wording

| Condition | Query text (the exact text used) |
|---|---|
| Grab original | “Point to the part you would grab.” |
| Grab phrases 1–3 | “Point to the part you would grasp.”; “Identify the region suitable for gripping.”; “Locate where you would grip the object.” |
| Pour original | “Point to the part you would use to pour.” |
| Pour phrases 1–3 | “Point to the part used for pouring.”; “Locate the region from which liquid is poured.”; “Identify the pouring region.” |
| Wrap original | “Point to the part you can wrap your hand around.” |
| Wrap phrases 1–3 | “Identify the region you would wrap your fingers around.”; “Locate the part that provides a wrapping grip.”; “Point to the region designed to be held by wrapping your hand around it.” |
| Accommodate original | “Point to the part of the object that can hold something.” |
| Accommodate phrases 1–3 | “Identify the region used to contain or store items.”; “Locate the area that can accommodate contents.”; “Point to the part where you would place or keep something.” |
| Unlock original | “Point to the part you would use to unlock this object.” |
| Unlock phrases 1–3 | “Locate the part used to open the object.”; “Identify the component that lets you open or release it.”; “Point to the part you would operate to gain access.” |
| Paraphrase average | Pointwise average of the three paraphrase score maps. |

## Paired aIoU change from the original query

Changes are percentage points; intervals are paired bootstrap 95% intervals across the selected objects.

| Affordance | Phrase 1 | Phrase 2 | Phrase 3 | Paraphrase average |
|---|---:|---:|---:|---:|
| grab | +0.11 [-1.85, +2.70] | -0.17 [-2.81, +2.72] | -1.33 [-3.87, +1.75] | -0.74 [-2.97, +2.11] |
| pour | +0.91 [+0.06, +1.76] | +0.81 [-0.78, +2.77] | -5.40 [-8.77, -2.38] | -1.59 [-2.99, -0.39] |
| wrap | +0.11 [-0.83, +0.92] | -1.68 [-2.49, -0.89] | +0.44 [-0.29, +1.20] | -0.49 [-0.99, -0.10] |
| accommodate | +1.52 [-0.73, +4.26] | +1.37 [-0.27, +3.84] | +1.68 [-0.24, +4.39] | +1.13 [-0.47, +3.48] |
| unlock | +0.90 [-0.59, +2.65] | +0.52 [-0.64, +2.06] | +0.94 [-0.23, +2.46] | +0.73 [-0.50, +2.26] |

## Interpretation

Prompt wording effects differ by affordance in this pilot. For pouring, phrase 1 (“Point to the part used for pouring”) was +0.91 points, while phrase 3 (“Identify the pouring region”) was −5.40 points; the three-phrase mean was −1.59 points. For wrap, phrase 2 and the phrase average were lower than the original. Grab wording changes were small on this fresh set. Accommodate and unlock paraphrases had higher mean aIoU, but their paired intervals include zero.

This is consistent with the earlier pouring and grab pilots in one useful way: no single paraphrase or averaging rule improves every affordance. It supports investigating affordance-specific prompt sensitivity, but it does not establish that wording reliably causes these changes.

## Limits

These are small deterministic subsets chosen because their ground-truth regions are present; they are not random or fully representative test samples. The bootstrap intervals are descriptive, apply only to the selected objects, and are not corrected for multiple comparisons. Region sizes vary widely (3 to over 1,800 points), and some labels/categories are concentrated in a few object types. The unofficial Affogato-pretrained model is being evaluated on OpenAD data, so this is a cross-dataset test. Prompts have not been human-validated. The run used CPU because MPS was unavailable; all E12 conditions share this device, but small numeric differences from earlier MPS runs are possible. Affostruction has not been run.

## Artifacts

- `run.json`: all prompts, label IDs, selected objects, metrics, paired intervals, and selection rules.
- `per_object_heatmaps.npz`: point coordinates, ground truth, and predicted maps.
- `mean_metrics.csv` and `per_object_metrics.csv`: scores in tabular form.
- `per_affordance_aiou.png`: plot of mean aIoU by affordance and query condition.
- `run_multiaffordance_prompt_study.py` and `summarize_multiaffordance_prompt_study.py`: scripts.
