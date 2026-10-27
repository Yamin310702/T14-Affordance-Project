# E15: threshold calibration and validation transfer

**Status:** complete on MPS. The goal was to set an action-specific cutoff from OpenAD training objects, then apply it unchanged to the positive/negative E14 validation objects for both original and selected paraphrase prompts. Train and validation shape IDs are disjoint. The prompts and model weights were fixed.

## Result: the calibration objective collapsed to empty predictions

For each affordance, the threshold maximized balanced mean object IoU on 40 positive and 40 negative training objects. On a negative object, predicting no points receives IoU 1.0. The optimizer therefore found the all-background solution for four of the five actions; on unlock it found almost the same result. This is a degenerate calibration result, not a useful deployment threshold.

| Affordance | Chosen threshold | Train positive objects detected at cutoff | Test positive original/paraphrase detected | Test negative original/paraphrase flagged |
|---|---:|---:|---:|---:|
| grab | 0.82 | 0/40 | 0/20; 1/20 | 0/20; 0/20 |
| pour | 0.83 | 0/40 | 0/20; 0/20 | 0/20; 0/20 |
| wrap | 0.73 | 0/40 | 0/20; 2/20 | 1/20; 1/20 |
| accommodate | 0.73 | 0/40 | 0/20; 1/20 | 2/20; 4/20 |
| unlock | 0.75 | 1/40 | 0/20; 0/20 | 0/20; 0/20 |

The selected cutoff exceeded the maximum original-query score on all E14 positive objects for every affordance. Consequently, the original query detected **0 of 20 positive objects for each action** at the calibrated cutoff. The selected paraphrase detected only one grab object, two wrap objects, and one accommodate object; it detected none for pour or unlock. Negative-object alarms were low because the cutoff suppressed nearly all predictions.

## Interpretation

The chosen threshold trades away nearly all target recall to suppress false alarms. That means the E14 increase in accommodate false alarms at 0.5 cannot be fixed simply by setting a high threshold without also losing target regions. The model scores do not cleanly separate the annotated positive and negative objects under this calibration objective. Threshold-free ranking metrics such as aIoU/AUC remain more informative for this study, while this cutoff result warns that point masks are difficult to obtain with a single threshold.

This experiment exposed a metric-design problem: balanced macro IoU rewards empty predictions on negative objects, and on this sample that yields 0.50 even though positive recall is zero. The procedure and result are preserved; do not treat these cutoffs as recommended settings. E14 was analyzed earlier, so this is exploratory threshold transfer rather than an untouched final test.

## Next experiment

If thresholded masks matter to the proposal, set an operating rule in advance (for example, maximize positive recall while keeping negative-object alarms under a stated limit), calibrate it on training objects, and evaluate it on a separate, not-yet-inspected object split. Report the full precision-recall and false-alarm curve as well as the selected operating point. The current result should be described as a failed attempt to find a useful cutoff with balanced macro IoU.

## Artifacts

- `run.json`: prompts, sampled training objects, calibrated cutoffs, test metrics, and run metadata.
- `calibration_predictions.npz`: training calibration maps and ground truth.
- `threshold_diagnostics.csv`: cutoffs and detection counts.
- `run.log`, `run_e15_threshold_calibration.py`, and `summarize_e15_threshold_calibration.py`: log and scripts.
