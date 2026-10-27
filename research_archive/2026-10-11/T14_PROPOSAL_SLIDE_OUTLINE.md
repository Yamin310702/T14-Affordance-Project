# T14 proposal: slide outline and experiment to propose

Drafted 2026-10-11 from the Week 7 proposal guideline and the archived OpenAD pilot results. This is a content outline for an 8-minute English presentation; add team/member details and the paired point-cloud figures before making the final deck.

## Recommended proposal direction

**Working title:** Robust Open-Vocabulary 3D Affordance Localization through Prompt Ensembling

**Research question:** On partial point clouds, does averaging several fixed, meaning-equivalent text prompts improve point-wise affordance localization over a single canonical query, and does the effect vary by affordance or increase false alarms?

**Falsifiable hypothesis:** With the OpenAD point backbone and CLIP text encoder frozen, averaging five meaning-equivalent prompt embeddings per held-out affordance will increase the 18-affordance macro point-wise IoU over a single canonical held-out query on the predeclared evaluation objects. The gain will vary by affordance and may raise false alarms on objects without the target affordance. The hypothesis is unsupported if the paired object-level 95% interval for the primary mIoU change includes zero or the observed change is non-positive.

**One intervention:** Replace each held-out affordance's single text embedding with the L2-normalized mean of five prewritten, meaning-equivalent prompt embeddings. Do not change point features, checkpoint weights, input points, labels, decision rule, or evaluation code. Compare prompt counts 1, 3, and 5 as the only planned ablation; K=5 is the primary condition and must be selected before looking at evaluation outcomes.

## Eight-minute, ten-slide story

| Slide | Time | Content to show |
|---|---:|---|
| 1. Title | 0:15 | T14, working title, team number and names. State the task in one phrase: predict actionable parts on a 3D point cloud from a text query. |
| 2. Problem | 0:55 | Input: 1,000–2,000 point XYZ samples and an affordance query. Output: a score/label for each point. Show one object under two queries producing different point regions. Give one short motivation: partial observations can hide actionable geometry. |
| 3. Baseline and setup | 0:55 | OpenAD as the public pretrained point-text baseline; checkpoint frozen, CLIP text features cached, inference evaluator adapted for Apple MPS. Identify the partial-view 3D-AffordanceNet data and distinguish measured MPS evaluation from the paper's original CUDA environment. Put the checkpoint hash and source/data details in appendix. |
| 4. Baseline evidence | 0:55 | Same-checkpoint, same-input paired query-vocabulary table. Partial views: training-name vs held-out-query mIoU is 41.45% vs 12.53% over 19 classes and 40.39% vs 10.77% over 18 affordances. Full shapes: 36.18% vs 9.76%, and 35.18% vs 7.87%. State clearly that this is query-vocabulary sensitivity, not proof of unseen-concept generalization. |
| 5. Observed failure | 0:55 | Show the saved Vase sample 452 visualization and actual point scores: target `pour` has 563 ground-truth points, none are correctly predicted, and every target point is assigned `none`; mean `pour` score on target points was 0.0584 vs 0.6440 for `none` in the matching failure analysis. Label these as one observed example, not an aggregate. |
| 6. Research question | 0:45 | Ask the prompt-ensemble question above. Connect it to the failure: point geometry alone may not explain the collapse; the text query participates in the point-text competition. Call this a possible explanation to test, not an established cause. |
| 7. Proposed method | 1:10 | Keep OpenAD and CLIP frozen. For each held-out query, encode five fixed equivalent phrasings, average and normalize their text embeddings, and rescore the exact same points. Primary comparison: K=5 ensemble vs K=1 canonical query. No k-NN smoothing or head fine-tuning in this experiment. |
| 8. Controls, metrics, and ablation | 1:05 | Same object IDs, point samples, checkpoint, class slots, `none` query, and argmax rule across conditions. Primary metric: global point-wise IoU macro over 18 affordances, excluding `none`; report 19-class official-style mIoU separately. Secondary: per-class IoU, AUC, negative-object false-positive point fraction and any-alarm rate. Paired bootstrap by object. Ablation: K=1, K=3, K=5. Prewrite/freeze prompts without using evaluation labels or scores. |
| 9. Schedule and risks | 0:40 | Weeks 7–8: lock prompts and split, implement embedding averaging, smoke test. Weeks 9–10: run fixed MPS evaluation and ablation. Weeks 11–12: paired analysis, class-wise errors and visualizations. Weeks 13–14: robustness checks and writing. Week 15: final presentation/report. Main risk: the available labeled split has already been used for baseline/pilot analysis; secure a fresh/object-disjoint evaluation set if available, otherwise label follow-up results exploratory and state this limitation. Fallback: complete the controlled validation study without claiming independent test generalization. |
| 10. Takeaway | 0:25 | “OpenAD is sensitive to query wording; one preselected `unlock` prompt helped but the effect did not transfer uniformly. We will test whether a fixed prompt ensemble improves average localization while measuring false alarms.” |

Total: 8:00. Keep detailed per-class tables, other E18 results, resource numbers, and method caveats in appendix slides rather than adding more main-talk content.

## Evidence and wording guardrails

- Label the current work **public OpenAD checkpoint, MPS-adapted evaluation**. E17 and E18 are completed pilot/validation analyses; they are not Affogato or Affostruction reproductions.
- MPS reproduced the CPU headline scores: E17 global mIoU changed by at most 0.0054 percentage points. The MPS runs were faster on this machine, but this is not a general hardware benchmark. See the E17 and E18 device-comparison notes.
- For the required one-input cost, report the profiled single cold forward on 2,048 points: **0.5456 s**, after model/data setup and without warm-up. The measured 1.295 GiB is a 5 ms sampled high-water Metal-driver allocation estimate on unified memory, not a guaranteed physical peak VRAM. Include the current tensor allocation separately only if there is room in the appendix.
- E18's `unlock` selected paraphrase reached 12.36% global IoU vs 0.09% for its original sentence, but the negative-object any-alarm rate increased from 0.34% to 1.61%. Report both sides of this trade-off.
- Use “training-name” and “held-out-query vocabulary.” Avoid “unseen affordance” or “unseen concept” unless the split definition supports that stronger claim; several paired terms are synonyms.
- Affogato informed exploratory prompt selection. Affostruction has not been run on this Mac; cite it only as related work/future comparison, not as a result.
- Proposal deadline from the guideline: submit the slide PDF to e-Class by **10:59 AM KST, October 15, 2026**. Use the required filename pattern `teamNN_T14_proposal.pdf`; add the editable source if useful.

## Source records

- [Proposal completion guide](PROPOSAL_COMPLETION_GUIDE.md)
- [OpenAD seen/held-out query results](seen_query_comparison/RESULTS.md)
- [E17 CPU/MPS comparison](seen_query_comparison_mps/DEVICE_COMPARISON.md)
- [E18 prompt transfer results](E18_openad_prompt_transfer/RESULTS.md)
- [E18 CPU/MPS comparison](E18_openad_prompt_transfer/mps/DEVICE_COMPARISON.md)
- [One-input resource profile](openad_sample452_mps_profile.json)
- [Proposal baseline audit](PROPOSAL_BASELINE_AUDIT.md)
