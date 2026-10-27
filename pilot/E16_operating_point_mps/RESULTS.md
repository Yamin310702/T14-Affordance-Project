# E16: accommodation prompt under a false-alarm cap — MPS rerun

**Status:** completed on Apple MPS. **Model:** unofficial Espresso-3D reimplementation with Affogato-pretrained weights. **Primary question:** can either locked accommodation prompt retain positive-object localization while keeping negative-object alarms at or below 10%?

## Design

Accommodation was selected as the single primary affordance because E14 showed the largest prompt-related false-alarm increase for this action. The original prompt and E13-selected paraphrase were fixed before E16. Calibration used 40 positive and 40 negative OpenAD training shapes; original-query scores were reused from E15 (MPS), and paraphrase calibration scores were computed on MPS. The decision rule was fixed in advance: among thresholds with no more than 10% negative-object alarms, maximize average per-object point recall. Prompt-specific cutoffs were frozen and evaluated on 20 positive and 20 fresh negative OpenAD validation shapes. Test IDs are disjoint from E09–E14 and calibration IDs.

A negative object counts as an alarm if any point score reaches the cutoff. The 20+20 test is stratified by ground-truth presence and does not estimate natural prevalence. The model was not trained on OpenAD; this is exploratory cross-dataset testing, not an official Espresso or Affogato benchmark.

## Main result: no useful cutoff met the 10% rule

| Prompt | Cutoff | Calibration positive recall | Calibration negative alarms | Test positive recall | Test negative alarms |
|---|---:|---:|---:|---:|---:|
| Original: “Point to the part of the object that can hold something.” | 1.00 | 0% | 0/40 | 0% (0/20 detected) | 0/20 |
| Selected paraphrase: “Point to the part where you would place or keep something.” | 1.00 | 0% | 0/40 | 0% (0/20 detected) | 0/20 |

Both prompts meet the alarm cap only by selecting a cutoff that detects no positive test objects. Neither has a useful operating point under this rule on these data.

## What the scores still show

On fresh positive objects, Espresso’s threshold-averaged aIoU was **5.99%** for the selected paraphrase and **3.64%** for the original. The paired change was **+2.35 percentage points** (object bootstrap 95% interval **+0.26 to +5.33 pp**, 20 objects). AUC was **42.4%** for the paraphrase and **38.4%** for the original, low for both.

At the uncalibrated diagnostic cutoff of 0.5, the original flagged 1/20 positive and 4/20 negative objects; the paraphrase flagged 8/20 positive and 19/20 negative objects. The paraphrase highlighted more positive objects but also fired on nearly every negative object. These 0.5 values are descriptive only and were not used to select E16 cutoffs.

![Calibration and held-out alarm–recall curves](operating_curve.png)

## CPU–MPS consistency

The same 40 positive and 40 negative calibration shapes and the same 20+20 fresh test shapes were run in the earlier CPU E16 and this MPS E16. All 400 shared arrays have matching IDs and labels. Across test score maps, maximum absolute CPU–MPS difference was **6.29×10⁻⁶** for the original wording and **2.44×10⁻⁶** for the paraphrase. Calibration paraphrase maps differed by at most **2.80×10⁻⁶**. All maps were `allclose` at absolute and relative tolerance 1×10⁻⁵. Both devices selected threshold 1.00 for each prompt and produced the same displayed metrics and alarm counts. The scientific interpretation is unchanged by device.

## Interpretation and next step

The selected wording gives a small exploratory improvement in threshold-averaged localization, but the model scores do not separate present from absent accommodation affordances well enough for a 10% false-alarm rule. Frame the proposal as a localization-versus-false-alarm trade-off; do not claim the paraphrase is uniformly better. Before another threshold test, agree on a task-justified false-alarm cap with the professor and continue to report the full calibration and held-out curves. Do not choose a new cap by optimizing on the held-out test set.

## Artifacts

- `run.json`: MPS run metadata, prompts, object IDs, cutoffs, and metrics.
- `predictions.npz`: calibration and fresh validation score maps, labels, and XYZ points.
- `summary.csv`, `thresholds.csv`, `threshold_curve.csv`, `operating_curves.csv`, `test_per_object.csv`: tabular results.
- `operating_curve.png`: calibration and held-out alarm–recall curves.
- `run_e16_operating_point.py`, `summarize_e16_operating_point.py`: runner and summarizer. The terminal output was shared in the conversation; `run.json` is the structured execution record.
- `DEVICE_COMPARISON.md`: CPU–MPS numeric comparison.
