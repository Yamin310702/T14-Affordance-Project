# E18 — fixed prompt transfer to OpenAD

**Status:** complete. **Model:** released OpenAD PN2 partial-view checkpoint, frozen. **Device:** CPU (MPS was unavailable in the active Python process). **Run time:** 940.7 seconds for model inference over 1,941 object groups.

## Main finding

The prompt selected earlier for **unlock** transferred to OpenAD; the other four selected paraphrases did not produce a comparable localization gain. This supports an affordance-specific prompt effect, not a rule that longer or paraphrased queries generally improve OpenAD.

For unlock, the selected prompt, “Point to the part you would operate to gain access,” raised global point-wise IoU to **12.36%**, compared with **0.09%** for the original sentence (“Point to the part you would use to unlock this object”) and **3.65%** for the canonical one-word query `unlock`. On positive objects, mean object IoU rose from **0.18%** to **8.58%**; the paired shape-level gain was **+8.40 percentage points** (95% bootstrap interval **+5.69 to +11.42**, 198 positive shapes). The prompt also raised mean false-positive point fraction on negative objects from **0.0016%** to **0.0396%**; the share of negative objects with any false alarm rose from **0.34%** to **1.61%**. This is a localization gain with a measurable, small absolute false-alarm cost.

## Results by affordance

Global IoU is point-weighted over all views; object IoU is averaged across positive shapes. False-positive fraction is averaged across shapes whose ground truth lacks the affordance. Percent values are shown.

| Held-out label | Canonical word global IoU | Original sentence global IoU | Selected paraphrase global IoU | Positive-shape IoU change, paraphrase − original (95% paired interval) | Negative-shape false-positive fraction, original → paraphrase |
|---|---:|---:|---:|---:|---:|
| grab | 1.38 | 0.00 | 0.00 | 0.00 pp [0.00, 0.00] | 0.0000 → 0.0000 |
| accommodate | 0.00 | 0.00 | 0.00 | 0.00 pp [0.00, 0.00] | 0.0000 → 0.0000 |
| unlock | 3.65 | 0.09 | **12.36** | **+8.40 pp [+5.69, +11.42]** | 0.0016 → 0.0396 |
| wrap | 7.90 | 0.00 | 0.09 | +0.05 pp [0.00, +0.12] | 0.0000 → 0.0113 |
| pour | 8.54 | 0.00 | 0.00 | +0.001 pp [0.000, +0.003] | 0.0000 → 0.0000 |

On positive objects, mean object ROC AUC for unlock was **84.20%** with the original sentence and **88.38%** with the selected paraphrase. The original sentence had nonzero AUC despite near-zero argmax IoU: it ranked target points above background to some extent, but the target class almost never won the 19-class competition. This illustrates why IoU and AUC answer different questions.

Grab and accommodate remained at zero target IoU for both sentence prompts. Wrap and pour changes were negligible by comparison with unlock; their paired intervals include zero. No smoothing, prompt ensembling, threshold selection, or training was added.

## Protocol and limits

E18 used the five original/selected prompt pairs fixed by E12/E13, without looking at OpenAD prompt scores. It compared the canonical held-out word, the original sentence, and the selected paraphrase. For each target affordance, only that class's text was changed; the other 18 class queries stayed fixed. Each point cloud's geometric features were computed once and reused for the paired text conditions.

The test pool contains **1,941 shapes and 7,764 partial views** from the official validation split. The 344 shapes used in E09–E13 were excluded, leaving unseen object IDs relative to those earlier prompt studies. Positive/negative shape counts were grab 81/1,860; accommodate 213/1,728; unlock 198/1,743; wrap 26/1,915; pour 34/1,907. The official dataset provides no separate held-out test split in this local setup, so E18 is a prior-study-disjoint validation diagnostic, not an official test result. Bootstrap intervals resample shapes, not views or points. Five affordances were tested and the intervals are not multiplicity-adjusted.

Prompts were selected using unofficial Espresso-3D with Affogato-pretrained weights, then transferred to OpenAD. The checkpoint was not changed. MPS was unavailable to the active process, so this run used CPU. The result motivates testing an affordance-aware prompt strategy, but it does not show that these prompts generalize to other datasets, other affordances, or real robot success.

The full MPS rerun reproduced the unlock paraphrase effect and its false-alarm trade-off; see the [CPU/MPS comparison](mps/DEVICE_COMPARISON.md) and saved MPS per-object outputs.

## Artifacts

- `PROTOCOL.md`: frozen study question, prompts, metrics, and leakage controls.
- `run.json`: device, checkpoint hash, selected/excluded counts, results, and bootstrap intervals.
- `per_affordance_summary.csv`: aggregate metrics for all 15 affordance-condition pairs.
- `per_object_metrics.csv`: per-shape outcomes for localization, AUC, and false alarms.
- `object_manifest.csv`, `selected_test_shape_ids.txt`, `excluded_prior_shape_ids.txt`: exact evaluation and exclusion records.
- `run.log`: progress and completion log.
- `pilot/run_e18_openad_prompt_transfer.py`: reproducible runner.
