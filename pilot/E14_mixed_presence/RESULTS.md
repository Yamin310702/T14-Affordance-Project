# E14: mixed-presence object evaluation

**Status:** complete on MPS. **Model:** unofficial Espresso-3D with Affogato-pretrained weights. **Data:** OpenAD validation partial views. **Design:** for each affordance, 20 objects were randomly selected from those with a positive ground-truth region and 20 from those with no ground-truth region. The same fixed original and E13-selected paraphrase were tested. Objects from E09–E13 were excluded. This stratified sample measures both localization and false alarms; it does not estimate natural prevalence.

## Positive objects: localization

Each affordance has 20 positive objects. Changes are selected paraphrase minus original aIoU; the intervals are paired bootstrap 95% intervals.

| Affordance | Original aIoU | Selected paraphrase aIoU | Change (pp) | 95% interval (pp) |
|---|---:|---:|---:|---:|
| grab | 12.81% | 14.37% | +1.56 | [-1.94, +5.74] |
| pour | 8.33% | 8.80% | +0.47 | [-0.03, +0.94] |
| wrap | 11.66% | 12.07% | +0.41 | [-0.48, +1.22] |
| accommodate | 1.42% | 1.46% | +0.04 | [-0.49, +0.69] |
| unlock | 3.67% | 2.94% | -0.73 | [-2.10, +0.43] |

## Negative objects: false alarms

For objects with no target points, aIoU and AUC are undefined. The “alarm” metric is the share of objects with at least one score ≥0.5. The “positive points” metric is the average share of points with score ≥0.5. This threshold is a diagnostic, not calibrated.

| Affordance | Query | Negative objects flagged | Mean points flagged |
|---|---|---:|---:|
| grab | Original | 15% (3/20) | 0.029% |
| grab | Selected paraphrase | 15% (3/20) | 0.012% |
| pour | Original | 20% (4/20) | 0.491% |
| pour | Selected paraphrase | 20% (4/20) | 5.720% |
| wrap | Original | 30% (6/20) | 1.885% |
| wrap | Selected paraphrase | 40% (8/20) | 2.402% |
| accommodate | Original | 30% (6/20) | 1.665% |
| accommodate | Selected paraphrase | 65% (13/20) | 11.172% |
| unlock | Original | 10% (2/20) | 0.022% |
| unlock | Selected paraphrase | 0% (0/20) | 0.000% |

## Interpretation

On the positive objects, the selected paraphrase had small or uncertain localization changes across all five affordances; each paired aIoU interval includes zero. On negative objects, the accommodate paraphrase raised the 0.5-threshold alarm rate from 30% to 65%, while wrap rose from 30% to 40% and pour stayed at 20%. Grab stayed at 15%; unlock decreased from 10% to 0%. The accommodated-region phrase therefore appears prone to false alarms under this threshold on this sample, despite its higher aIoU in earlier positive-only tests.

This result shows why both presence and absence cases matter. It weakens any broad claim that choosing the best paraphrase from positive-only data improves real-world behavior. The findings remain preliminary: only 20 objects per stratum, deterministic hash selection, one dataset and model, class-balanced sampling rather than natural prevalence, and an uncalibrated threshold. Multiple comparisons are not adjusted.

## Prompts

The original and selected paraphrase text for each affordance is recorded in `run.json`; they are the exact E13 pairs.

## Saved files

- `run.json`: sample IDs, prompts, model, metrics, paired intervals, and eligibility counts.
- `predictions.npz`: XYZ, labels, and score maps.
- `mean_metrics.csv` and `per_object_metrics.csv`: tabular results.
- `paired_effects.png`: paired changes on positive and negative groups.
- `run.log`, `run_e14_mixed_presence.py`, and `summarize_e14_mixed_presence.py`: run log and scripts.
