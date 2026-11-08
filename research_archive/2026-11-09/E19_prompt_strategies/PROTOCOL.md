# E19: three fixed query strategies on frozen OpenAD

Protocol prepared 2026-11-09, before examining E19 scores. This is the next implementation step of the current Team 07 proposal. It supersedes the K=1/3/5 plan in older handoff notes; those records remain historical.

## Question and fixed choices

Does averaging three fixed paraphrase probability maps improve 18-action macro IoU versus label-aligned canonical query names, and what happens to false alarms?

The user selected label-aligned names as the canonical baseline on 2026-11-09. Use the exact audited training-name class order, including the original `displaY` spelling. Do not use the replacement-name E17 vocabulary as the primary baseline. All 18 slots change together in the paraphrase conditions; this differs from E18's one-slot-at-a-time intervention. No claim of new-affordance generalization follows from this comparison.

- **A:** 18 canonical queries and the fixed `none` competitor.
- **B:** first prewritten paraphrase for each of the same 18 class slots and `none`.
- **C:** arithmetic average of three complete 19-class softmax probability maps, one per paraphrase vocabulary. Average all channels, including `none`, so the mean remains a normalized distribution. Take argmax after averaging. Never average text embeddings or already discretized masks.

`prompts.json` contains the 54 fixed phrases. B is fixed as phrase 1, not whichever phrase performs best. These are intended semantic paraphrases; semantic equivalence is an assumption, especially for dataset-specific categories such as `layable`, `support`, and `wrap_grasp`. Do not revise phrases after inspecting E19 scores and retain the same experiment identity. Any revision requires a separate version and disclosure.

## Geometry, model, data and compute

Use the released PN2 partial checkpoint for both partial input and the full-shape control, matching the prior comparison. Freeze all model weights. Evaluate with the adapted local OpenAD cosine scoring path and learned logit scale as implemented, without changing it to exp(logit_scale). Preserve the model's preprocessing.

For each shape compute geometry features once, sharing them across every query condition. Seed point-backbone sampling by a deterministic hash of shape ID and global seed. Never process A/B/C with independent point sampling. Sort shape IDs before any smoke-run limit. Save hashes of the checkpoint, runner, prompt file and input dataset files. Refuse to overwrite any output directory. Explicit device requests must not silently fall back.

Primary input: all local official validation partial views, grouped by object. Control: full shapes for the same object IDs. Use the validation set only: prior experiments have already examined this data, including E17/E18. The study is **exploratory reused validation**, not a fresh test set, even if individual prompts are newly fixed. No tuning or training is added. A limited smoke run is solely an implementation check and must not be reported as an 18-action benchmark.

## Outcomes and uncertainty

Primary outcome: aggregate point-wise IoU for each action, followed by the macro average of the 18 affordance slots (exclude `none`). Primary contrast C minus A. Secondary contrast B minus A. For each bootstrap draw resample whole shape IDs, sum TP/FP/FN over all their views, and recompute class IoU and macro IoU. Do not average view-level IoU or bootstrap individual points. Use 2,000 paired resamples, seed 20261109. Report defined-action counts; an absent-union action is undefined and excluded, not automatically scored as zero. On a smoke set this can differ by condition, so its macro values are not a benchmark.

For each action, classify an object as label-negative only when that action is absent from **all stored full-shape validation annotations**. This is annotation-negative, not external verification of physical impossibility. Partial-view absence alone is insufficient. Stop if a partial label contradicts a full-shape negative claim.

For those label-negative objects report (a) fraction of evaluated points assigned to the action and (b) share of objects with at least one assigned point. Use the same fixed 19-way argmax decision rule in A/B/C, with no tuned threshold. Report paired object-level differences and percentile intervals for both false-alarm measures. Report per-action denominators. The strict any-point alarm definition is sensitive to point/view count; compare only paired identical inputs. Action-level intervals are exploratory and not multiplicity-adjusted.

No universal reliability claim follows from macro IoU alone. Report trade-offs, regressions, and null findings. The current proposal predicts no reliable positive C-minus-A mean-IoU gain. A positive interval does not establish external generalization on reused validation.

## Initial implementation scope

The initial runner saves object-level TP/FP/FN, global per-action IoU, 18-action macro IoU, both negative-object alarm measures, and paired intervals. AUC is not yet computed; no AUC result is claimed. Raw per-point probabilities and qualitative figures are not yet exported. Complete inference must follow implementation verification; successful smoke runs do not establish the hypothesis.

## Reproduce

From the project root, using the existing OpenAD environment:

```sh
/opt/homebrew/Caskroom/miniforge/base/envs/openad/bin/python pilot/run_e19_prompt_strategies.py --split partial --device cpu --max-shapes 3 --bootstrap 200 --output-dir /private/tmp/e19_partial_check
/opt/homebrew/Caskroom/miniforge/base/envs/openad/bin/python pilot/run_e19_prompt_strategies.py --split full --device cpu --max-shapes 3 --bootstrap 200 --output-dir /private/tmp/e19_full_check
```

After verification, omit `--max-shapes` and retain the default 2,000 bootstrap draws for complete runs. Use separate new directories, for example `partial_cpu` and `full_cpu` under this study folder. On a machine with MPS available, explicitly use `--device mps` and a separate destination. `--repo` and `--checkpoint` support collaborator-specific paths.
