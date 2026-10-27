# E18 — CPU/MPS reproducibility comparison

**Status:** the full MPS run completed on 1,941 prior-study-disjoint validation shapes (7,764 partial views), using the same frozen checkpoint, prompts, shape IDs, and bootstrap seed as the CPU run. The raw JSON, CSV files, and selected/excluded shape-ID lists are saved in this folder.

## Result

The main prompt-transfer finding reproduces: the preselected `unlock` paraphrase substantially improves localization over the original sentence, with a measurable false-alarm cost. On MPS, its global IoU is **12.3621%**, compared with **0.0927%** for the original sentence and **3.6878%** for the canonical word `unlock`. The CPU values were 12.3593%, 0.0927%, and 3.6461%, respectively.

| Target | CPU selected paraphrase IoU | MPS selected paraphrase IoU | CPU positive-shape IoU gain vs original (95% interval) | MPS positive-shape IoU gain vs original (95% interval) |
|---|---:|---:|---:|---:|
| grab | 0.0000% | 0.0000% | 0.00 pp [0.00, 0.00] | 0.00 pp [0.00, 0.00] |
| accommodate | 0.0000% | 0.0000% | 0.00 pp [0.00, 0.00] | 0.00 pp [0.00, 0.00] |
| unlock | 12.3593% | 12.3621% | +8.404 pp [+5.690, +11.416] | +8.406 pp [+5.691, +11.418] |
| wrap | 0.0916% | 0.0916% | +0.053 pp [0.00, +0.123] | +0.053 pp [0.00, +0.123] |
| pour | 0.0019% | 0.0019% | +0.001 pp [0.00, +0.003] | +0.001 pp [0.00, +0.003] |

For unlock, the MPS mean false-positive point fraction on negative shapes was **0.0396%** for the selected paraphrase versus **0.0016%** for the original sentence. The negative-shape any-alarm rate rose from **0.34%** to **1.61%**. These match the CPU conclusion and are part of the result, not an incidental caveat. The other four paraphrases did not show a comparable localization gain.

## Runtime and interpretation

CPU inference took 940.7 seconds and MPS took 569.2 seconds for this run. A few canonical-word metrics, notably `pour`, vary slightly between devices, but the paired prompt-effect estimates and intervals for the main `unlock` finding are nearly unchanged. This is consistent with small backend/numerical effects and does not change the interpretation. This is one device timing comparison, not a general benchmark.

E18 remains an exploratory validation transfer from prompts selected using unofficial Espresso-3D/Affogato experiments. The test objects were disjoint from E09–E13 prompt-study IDs, but this is not an official held-out test set, the five affordances were selected in advance, and the bootstrap intervals are not adjusted for multiple comparisons. The result supports an affordance-specific prompt effect for `unlock`; it does not establish broad prompt generalization or real robot success.

## Saved files

- `run.json`: device, run status, per-affordance metrics, paired shape bootstrap
- `per_affordance_summary.csv`, `per_object_metrics.csv`, `object_manifest.csv`
- `selected_test_shape_ids.txt`, `excluded_prior_shape_ids.txt`
- CPU comparison: [E18 results](../RESULTS.md)
- Runner: `pilot/run_e18_openad_prompt_transfer.py`
