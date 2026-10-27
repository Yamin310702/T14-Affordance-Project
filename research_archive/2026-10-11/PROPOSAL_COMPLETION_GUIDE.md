# T14 proposal completion guide

Updated 2026-10-11 from the saved OpenAD configs, baseline logs, and later prompt studies.

## Status of the audit items

| Audit item | Status | What to put in the proposal / what remains |
|---|---|---|
| Seen/unseen grouping | Partly resolved from the archived OpenAD configs | The config has 18 `train_affordance` names and 18 corresponding `val_affordance` query names in the same order, plus `none`. Existing E01/E05 open-vocabulary numbers use the validation query vocabulary. They are not a seen-query result. To report both groups, evaluate the same checkpoint and point clouds with the training-name vocabulary and the held-out validation-query vocabulary, then average the 18 affordance IoUs per group, excluding `none` from the affordance-only macro mean. Keep the original 19-class official mIoU as its own reported metric. Describe this carefully as seen vs held-out query vocabulary unless the instructor confirms that these are to be called unseen *affordances*; many pairs are semantically related/synonyms, so “unseen concepts” would overstate the split. |
| Memory | Complete with an explicit measurement caveat | Report the MPS sampled high-water Metal-driver allocation of 1,390,460,928 bytes (~1.295 GiB total process allocation) and label it a 5 ms sampled estimate, not physical peak VRAM. Report current tensor allocation after forward separately (~0.677 GiB). Apple Silicon uses unified memory. |
| Timing | Complete with an explicit scope | Report 0.5456 s for one synchronized, cold OpenAD forward on 2,048 points after model/data setup, with no warm-up. Call it a one-input profile, not steady-state latency or a benchmark. The older 0.335 s CPU / 0.874 s MPS figures have unclear timer boundaries and should not be compared. |
| Freeze one proposed protocol | Needs final lock | The older audit proposed k-NN smoothing. Later E08–E16 results make query wording the better-supported question for this proposal. Use one change only: compare the canonical held-out affordance query against one fixed, meaning-equivalent paraphrase per affordance on the same frozen OpenAD checkpoint. Use partial-view data as primary and full-shape as a control. Do not combine smoothing and prompt changes. |

## Candidate vocabulary mapping from the configs

The following mapping comes from the order of `train_affordance` and `val_affordance` in `research_archive/2026-10-05/code/openad/config/openad_pn2/partial_view_open_vocab_cfg.py` and the corresponding full-shape open-vocabulary config. It is a config-derived candidate split; preserve the original strings and confirm terminology against the instructor's intended split before calling the second column “unseen affordances.”

| Training vocabulary (seen query) | Validation vocabulary (held-out query) |
|---|---|
| grasp | grab |
| contain | accommodate |
| lift | raise |
| openable | unlock |
| layable | rest |
| sittable | take a seat |
| support | bear |
| wrap_grasp | wrap |
| pourable | pour |
| move | reposition |
| displaY | demonstrate |
| pushable | push |
| pull | drag |
| listen | hear |
| wear | clothe |
| press | thumb |
| cut | slice |
| stab | jab |
| none | none |

The order is aligned in the config, which makes a same-input query-vocabulary comparison possible. However, `none` is not an affordance and should be excluded from an 18-action macro average; retain it when reporting the code's original 19-class mIoU so the two measures are not confused.

## Recommended research question and locked comparison

### Seen-name vs held-out-query comparison — completed 2026-10-11

The same released PN2 checkpoint was evaluated on all 9,140 partial validation views and 2,285 full validation shapes using both aligned vocabularies. Geometry features were shared within each pair; only query names changed. The complete paired evaluation was run on CPU and rerun on MPS. The scores agree within 0.006 percentage points; see the [device comparison](seen_query_comparison_mps/DEVICE_COMPARISON.md). On partial views, training-name / held-out-name mIoU was **41.45% / 12.53%** over 19 classes including `none`, and **40.39% / 10.77%** over the 18 affordances excluding `none`. On full shapes, the corresponding scores were **36.18% / 9.76%** and **35.18% / 7.87%**. The partial held-out-query result also closely matches E05's earlier MPS result (12.536%).

This is evidence of query-vocabulary sensitivity in the released checkpoint, not evidence of novel-concept generalization: many aligned names are semantic aliases, and the training-name queries match the vocabulary used by the checkpoint. See the [full results and limitations](seen_query_comparison/RESULTS.md), including per-class tables and paired shape-level uncertainty analysis.

**Question:** On partial point clouds, do preselected meaning-equivalent prompts change OpenAD's point-level localization and false alarms compared with the held-out query word or its original sentence, and does the effect vary by affordance?

**Hypothesis:** Prompt wording has affordance-specific effects. A preselected paraphrase can improve localization for some affordances, but the gain will not be uniform and may increase false alarms for absent affordances. E18 supports this as a tentative claim: the selected `unlock` paraphrase improved IoU on prior-study-disjoint OpenAD validation objects, while the other four tested paraphrases did not show a comparable gain. The result remains exploratory and does not establish generalization beyond those labels and that validation split.

Lock these conditions before the test evaluation:

1. Keep OpenAD PN2 and its released checkpoint frozen. Do not add k-NN smoothing in this experiment.
2. Use identical object/view IDs and point samples for the canonical query and its paraphrase. Use development data only to choose prompts and any decision threshold; do not tune on the test set.
3. Make partial-view unseen/held-out queries the primary analysis. Report seen training-name queries and full-shape results separately as controls.
4. Use official point-wise IoU/mIoU as the primary metric and AUC as a secondary metric. For objects with no target region, separately report the predeclared false-alarm measure and threshold.
5. Compute paired differences by object, not by treating points or repeated views as independent. Preserve per-affordance rows and a few representative failure visualizations.

The prompt variants in E08–E16 were evaluated with unofficial Espresso-3D and Affogato-pretrained weights. They supplied E18's prompts, but their aIoU/AUC and false-alarm values are not OpenAD results. E18 is a separate, prior-study-disjoint OpenAD validation evaluation; keep the model/data caveats with both result sets.

## Immediate checklist

- [ ] Confirm with the instructor that the config-derived vocabulary split above matches their required seen/unseen definition.
- [x] Run training-name and held-out-query vocabularies on the same complete partial and full validation inputs on CPU and MPS; save per-class IoUs and compare devices.
- [x] Freeze five preselected equivalent prompt pairs from E12/E13 before OpenAD evaluation; save exact strings in E18's protocol.
- [x] Run one-class-at-a-time prompt substitutions on prior-study-disjoint OpenAD validation objects. This is not an official test-set result; see [E18](E18_openad_prompt_transfer/RESULTS.md).
- [ ] Make proposal slides that clearly label the released OpenAD checkpoint, the MPS E05 result versus this paired CPU comparison, actual measured values, one-input timing scope, sampled unified-memory estimate, and untested Affostruction.

## Saved evidence

- [OpenAD baseline audit](PROPOSAL_BASELINE_AUDIT.md)
- [Complete experiment evidence map](ALL_EXPERIMENTS_PROPOSAL_MAP.md)
- [Seen-name vs held-out-query comparison](seen_query_comparison/RESULTS.md)
- [Seen-query CPU/MPS reproduction](seen_query_comparison_mps/DEVICE_COMPARISON.md)
- [E18 CPU/MPS prompt-transfer reproduction](E18_openad_prompt_transfer/mps/DEVICE_COMPARISON.md)
- [Profile JSON](openad_sample452_mps_profile.json)
- [Baseline table](OPENAD_PROPOSAL_BASELINE_TABLE.csv)
- Open-vocabulary config files under `research_archive/2026-10-05/code/openad/config/openad_pn2/`
