# E18 — fixed prompt transfer to OpenAD

**Status:** completed on 2026-10-11. See [E18 results](RESULTS.md).

## Question and hypothesis

On partial point clouds, does a fixed, meaning-equivalent sentence prompt change OpenAD localization for five held-out query labels? The working hypothesis is that effects vary by affordance; some selected paraphrases may improve target IoU, while some may increase false alarms on objects without that affordance.

## Data and leakage controls

- Use every remaining object in the official OpenAD partial-view validation split after excluding shape IDs used by E09, E10, E11, E12, and E13. This removes prior prompt-study objects from the evaluation. Keep every partial view for each retained shape, and treat shape—not view or point—as the resampling unit.
- No OpenAD validation score or label is used to choose a prompt. The five paraphrases below were selected in E12 and frozen before the E13 Espresso confirmation. E18 transfers that fixed set to OpenAD.
- The result is a held-out-object diagnostic within the validation split, not a score on an official test split. E13 used Espresso/Affogato, so this is cross-model transfer of prompts, not an OpenAD prompt-tuning run.

## Fixed conditions

Use the released OpenAD PN2 partial-view checkpoint frozen. For each affordance, compare three vocabularies: (1) the canonical one-word held-out query; (2) the corresponding E13 original sentence; and (3) the E13-selected paraphrase. Replace only the target class text; keep all other class queries at their canonical held-out names, including `none`. Run the same point cloud through the geometry backbone once and score its features against all three vocabularies. Use argmax over the 19 classes, as in the released evaluator. No smoothing, fine-tuning, prompt averaging, or threshold tuning.

| Class index | Canonical held-out query | Original sentence | Frozen selected paraphrase |
|---:|---|---|---|
| 0 | grab | Point to the part you would grab. | Point to the part you would grasp. |
| 1 | accommodate | Point to the part of the object that can hold something. | Point to the part where you would place or keep something. |
| 3 | unlock | Point to the part you would use to unlock this object. | Point to the part you would operate to gain access. |
| 7 | wrap | Point to the part you can wrap your hand around. | Point to the region designed to be held by wrapping your hand around it. |
| 8 | pour | Point to the part you would use to pour. | Point to the part used for pouring. |

## Metrics and analysis

- Primary: global point-wise IoU for each target class across all retained partial views, with class labels fixed by the dataset.
- Secondary: positive-object IoU, per-object ROC AUC where both positive and negative target points occur, and false-positive point fraction plus any-alarm rate on objects whose ground truth lacks that affordance.
- Compare paraphrase with original sentence using paired bootstrap intervals over shape IDs (10,000 resamples, seed 20261011). Report canonical one-word queries as a diagnostic reference. Do not combine affordance metrics into a single claim or treat points/views as independent.
- Report all five affordances and the complete selection counts; interpret as exploratory because five affordances are tested and intervals are not multiplicity-adjusted.

## Expected output

Save the exact selected/excluded object manifest, immutable prompt strings, checkpoint hash, device, per-object metrics, per-affordance summary, and logs under this directory. If MPS is unavailable to the active Python process, run on CPU and label the device.
