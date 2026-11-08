# E19: current main-project experiment

Updated 2026-11-09. **Implementation and smoke verification complete. Full validation experiment not yet run.** No E19 improvement claim is supported yet.

This implements the current Team 07 proposal: frozen OpenAD, identical geometry, and three query strategies. The user selected the label-aligned canonical baseline, not the older replacement vocabulary. This replaces the older K=1/3/5 plan in the historical handoff.

## Fixed experiment

| Condition | Query/scoring strategy |
| --- | --- |
| A | Label-aligned canonical names |
| B | First prewritten paraphrase per action |
| C | Mean of three fixed 19-class paraphrase probability maps, then argmax |

The `none` text stays fixed. The complete probability distributions are averaged, including their `none` channels. No prompt is selected using E19 scores. All 18 query slots change together; E18 changed one target slot at a time, so the studies answer different questions.

Read [PROTOCOL.md](PROTOCOL.md) and [prompts.json](prompts.json) before running. The [runner](../../../pilot/run_e19_prompt_strategies.py) accepts collaborator-specific `--repo` and `--checkpoint` paths. Output directories must be new; existing records are never overwritten.

## Verified so far

- Five numerical tests passed: probability averaging before argmax, pooled IoU rather than object-IoU averaging, the `none` competitor, paired bootstrap invariance for identical predictions, and prompt/class-slot alignment.
- `smoke_partial/`: complete CPU inference on 3 shapes / 12 views.
- `smoke_full/`: complete CPU inference on the same 3 shapes.
- Both runs share the exact prompt and checkpoint hashes and matching shape manifests.
- `native_forward_verification.json`: canonical probabilities agree with the adapted upstream model forward pass on one shape / four partial views. Maximum absolute difference 2.98e-7; argmax labels identical.
- Full-shape annotation absence determines label-negative object/action pairs. Partial-view absence alone does not.
- MPS was unavailable in this Python process. These checks explicitly used CPU.

The smoke sets cover only a few action labels. Their macro scores and intervals are diagnostics, not an 18-action benchmark or hypothesis result. Do not compare them to the archived dataset-wide E17/E18 numbers.

## Run the complete study

From the project root, run sequentially using the existing OpenAD environment:

```sh
/opt/homebrew/Caskroom/miniforge/base/envs/openad/bin/python pilot/run_e19_prompt_strategies.py --split partial --device cpu --output-dir research_archive/2026-11-09/E19_prompt_strategies/partial_cpu
```

After that finishes:

```sh
/opt/homebrew/Caskroom/miniforge/base/envs/openad/bin/python pilot/run_e19_prompt_strategies.py --split full --device cpu --output-dir research_archive/2026-11-09/E19_prompt_strategies/full_cpu
```

Both commands use all eligible validation shapes and 2,000 paired object bootstrap draws. On the same dataset the two runs evaluate matching object IDs. Check `status: complete`, object counts, defined-action counts, and hashes in each `run.json` before interpreting results. On another computer change `--repo` or `--checkpoint`; on MPS use an explicit device request and a separate output directory.

## What the runner saves

- `run.json`: frozen prompts, hashes, device, dataset identity, scope, 18-action macro IoU, paired IoU intervals, per-action paired false-alarm differences and intervals.
- `shape_ids.txt`: exact object manifest.
- `object_counts.npz`: per-object, per-condition, per-class TP/FP/FN for paired reanalysis.
- `per_object_action.csv`: per-object localization counts and label-negative flags.
- `negative_object_action.csv`: annotation-negative false-positive point fractions and any-point alarms.
- `per_action_summary.csv`: per-action global IoU and false-alarm summaries.

AUC and per-point qualitative exports are not implemented in this initial runner. No AUC or visualization result is claimed. Full-run result interpretation and figures remain subsequent work.

## Claim limits

These are adapted OpenAD experiments on a validation set examined by earlier studies. Even a complete run remains exploratory reused validation. Semantic equivalence of the prewritten phrases is an assumption, not a measured fact. An annotation-negative pair is not independently verified physical impossibility. The strict any-point alarm rate is sensitive to point/view counts; only paired identical inputs should be compared.

Report C-minus-A and B-minus-A localization changes alongside false-alarm changes. Preserve negative findings and trade-offs. Do not tune prompts or thresholds against these results and then call the same evaluation held out.
