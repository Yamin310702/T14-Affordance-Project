# Background Suppression in 3D Affordance Prediction

**Task:** T14, 3D affordance prediction  
**Course:** CSE 59358-01, 3D Machine Intelligence, Fall 2026  
**Proposal direction:** a controlled study of the output decision in frozen OpenAD  
**Status:** measured baseline and observed failure; the proposed bias experiment has not run

## Abstract

Open-vocabulary 3D affordance prediction localizes object regions that support a text-described action. On partial point clouds, a model may suppress valid regions by assigning them to the background class. Our archived OpenAD PointNet++ evaluation achieved 12.5364% mean point-wise IoU over 19 classes on 9,140 partial validation views. An observed vase failure assigned all 563 annotated pour points to `none`, producing zero pour IoU. This proposal tests whether a single scalar adjustment to the `none` log score improves localization while limiting false positives. The point encoder, text encoder, and pretrained weights remain frozen. We will select the scalar using development objects from the official training split, then compare adjusted and unchanged predictions on identical saved validation inputs. We will report the released 19-class mIoU, foreground-only mIoU, recall, and background errors, with paired uncertainty estimates that group views by object. A gain is a hypothesis, not a completed result. The study aims to distinguish a decision-rule limitation from poor ranking among foreground affordances.

## 1. Problem and motivation

Given a point cloud of an object and an affordance query such as `pour`, predict which observed points support that action. The T14 guide permits pretrained point-text alignment with a frozen, cached text encoder and a small head or post-processing change. The proposed study changes only output post-processing.

The archived implementation jointly scores 18 affordance queries and `none`, then predicts the maximum-scoring class per point. We study this decision rule explicitly. An affordance score is a model output; it does not measure the probability of physical robot success. Localization on observed points also differs from reconstructing hidden surfaces.

## 2. Baseline and related work

[OpenAD (Nguyen et al., IROS 2023)](https://arxiv.org/abs/2303.02401) aligns point and language features for open-vocabulary affordance detection. Its [official repository](https://github.com/Fsoft-AIC/Open-Vocabulary-Affordance-Detection-in-3D-Point-Clouds) supplies code and pretrained models. Baseline status: **public**, with locally adapted MPS evaluation.

[Affogato (Lee et al., ECCV 2026)](https://arxiv.org/abs/2506.12009) introduces large-scale query-conditioned affordance data and Espresso models. [Affostruction (Park et al., CVPR 2026)](https://arxiv.org/abs/2601.09211) grounds affordances on complete geometry reconstructed from partial RGBD observations. These provide context for data scale and missing geometry. The current controlled experiment uses OpenAD and 3D-AffordanceNet. Archived unofficial Espresso results use a separate protocol and do not form a direct numerical comparison. Affostruction has not run locally.

## 3. Measured evidence

| Archived run | Setting | Recorded mIoU |
| --- | --- | ---: |
| E01 historical baseline | Full shape | 14.4121% |
| E01 historical baseline | Partial view | 12.4964% |
| E05 later MPS evaluation | All 9,140 partial validation views | 12.5364% |

The evaluator pools intersections and unions over points and views for each class, then averages the 19 class IoUs, including `none`. The E01 logs have less complete metadata than E05. Sampling and batch-size differences prevent treating E01 and E05 as identical reruns. Full/partial differences are descriptive and do not isolate the causal effect of occlusion. These are locally measured checkpoint results, not paper-reported scores.

**Observed failure:** archived partial validation sample 452 is a vase, shape ID `edb8e199485be4e968747a7f12307bdb`. In the saved E04 diagnostic, all 563 pour-labeled points receive `none`, and pour IoU is zero. The archived qualitative figure shows the annotation and empty pour prediction. A later profiled forward reproduces zero pour IoU. Keep any quoted score averages tied to their corresponding run, because stochastic sampling causes score values to vary.

**Plausible cause:** competition with `none` may conceal some useful foreground signal. **Alternative explanation:** the model may also rank the wrong foreground class highest. A scalar background adjustment cannot repair foreground ranking, so an unsuccessful result would be informative.

## 4. One research question

**Does a development-selected scalar adjustment to the `none` score improve OpenAD localization on partial point clouds without an excessive increase in background false positives?**

**Falsifiable hypothesis:** compared with unchanged OpenAD on identical validation inputs, the frozen adjustment increases 19-class point-wise mIoU and raises the background-point false-positive rate by no more than 5 percentage points.

Support requires both a paired object-bootstrap 95% confidence interval for the mIoU difference entirely above zero and the specified false-positive guardrail. A nonpositive gain, an interval crossing zero, or a guardrail violation leaves the hypothesis unsupported. Report the estimates even when the hypothesis fails. The 5 pp tolerance is a proposed research-design choice, not a validated deployment requirement.

## 5. Proposed method

For point i and class c, let s(i,c) be the frozen OpenAD log score. Define:

```
s_adjusted(i, none) = s(i, none) - b
s_adjusted(i, c)    = s(i, c)          for c != none
prediction(i)      = argmax_c s_adjusted(i, c)
```

At b = 0, this is the unchanged baseline. Subtracting b from a log score is equivalent, for argmax, to multiplying the background probability by exp(-b). The adjusted values need renormalization before they can be interpreted as a normalized distribution; the main experiment uses labels only. The adjustment preserves the ranking of all foreground classes. It can promote the strongest existing foreground prediction but cannot create new geometric evidence.

Candidate values: **b in {0, 0.25, 0.5, 1, 1.5, 2, 3}**. Use a single global scalar to keep the scope small and the interpretation clear. No class-specific biases, prompt changes, geometric smoothing, or training changes enter the primary experiment.

## 6. Comparison and controls

1. Save a deterministic, category-stratified development manifest of up to 200 unique official training objects. Keep every selected view grouped under its shape ID. Check train/validation IDs and point-label alignment. Describe the development subset and category counts exactly.
2. Evaluate the same replacement-query list used in the archived partial configuration. Cache frozen point scores and text features. Audit that cached and uncached outputs agree on a small fixed sample.
3. On development data, maximize the released 19-class mIoU subject to a background false-positive increase of at most 5 pp relative to b = 0. Break ties toward the smallest b. If no nonzero bias qualifies, select zero and report that outcome.
4. Freeze the choice before validation. Evaluate b = 0 and the selected b on exactly the same saved points and score arrays. Fix query ordering, checkpoint, label mapping, point count, normalization, class definitions, and metric implementation.
5. Reevaluate the official partial validation data, reporting per-class and aggregate results. All validation views have already been scored in earlier work. This is a predeclared exploratory reevaluation of previously examined validation data, not an untouched confirmatory test. Do not repeatedly retune b after looking at validation results.
6. Bootstrap unique object IDs, carrying all their views together and recomputing the aggregate metric for each paired resample. Use 2,000 resamples and a fixed bootstrap seed. Record the number of unique objects, not just views.

This evaluates object-level train/validation separation after verifying the official split. It does not establish generalization to unseen object categories unless a separate verified category-disjoint protocol is added. Checkpoint training membership and calibration membership also differ from evaluation membership and must be recorded.

## 7. Metrics and ablations

**Primary:** released 19-class mIoU, matching the baseline metric, including `none`.

**Secondary:** mean IoU over the 18 foreground classes, per-affordance IoU, foreground recall, none-class IoU, and background-point false-positive rate. The latter equals the fraction of annotated `none` points assigned any foreground class. Also report per-query negative-object alarms where annotations genuinely establish absence. Keep the point-level and object-level error definitions separate.

**Bias ablation:** plot performance and false positives against the candidate b values on development data. Any validation sweep is descriptive only and cannot select a new deployed bias.

**Full-cloud control:** apply the same locked b to the appropriate released full-cloud checkpoint and matching full-cloud evaluation. Compare b = 0 with b = selected within that setting. Do not interpret the full/partial comparison as an isolated causal test, since the checkpoints and input conditions may differ. Recover the full-checkpoint identity before reporting this control.

**Failure analysis:** inspect recovered regions, wrong foreground promotions, unchanged misses, and erosion of true background regions. Show both gains and regressions. Do not choose a new method using the final reported examples.

**Vocabulary:** the archived config maps training names to replacement phrases, such as `grasp` to `grab` and `pourable` to `pour`. New wording alone does not establish a new action concept. Map the professor's intended seen/unseen grouping explicitly before producing those aggregates. Label current results as the replacement-query setting until that mapping is verified.

## 8. Feasibility and reproduction

- Hardware: Apple M3 Pro, 18 GB unified memory, PyTorch MPS.
- Model: `OpenAD_PN2`, released partial checkpoint.
- Recorded source commit: `b082265ed085455a3fdc80b0d8d95dc51351e7b4`.
- Partial-checkpoint SHA-256: `cbcfe9ff70448d97a7047f1ade817645a9be29dd2782fba57776792a8210f2ae`.
- Profile environment: Python 3.10.21 and PyTorch 2.14.1.
- E05: seed 1, batch 1, 9,140 completed views, 2,416.95 seconds for the recorded dataset analysis. This elapsed time is not per-input inference latency.
- Profiled sample: 2,048 points, synchronized forward 0.5456 seconds after setup, one cold forward without warm-up.
- Memory: 1,390,460,928 bytes, approximately 1.295 GiB, observed high-water Metal-driver allocation with 5 ms sampling. After-forward current tensor allocation: 727,330,816 bytes, approximately 0.677 GiB. Neither is a CUDA-style guaranteed physical peak VRAM measurement.
- Local modifications are recorded in model files, test runner, and evaluator. Package the complete diff, environment, dataset manifest, and checkpoint identities rather than relying on the upstream commit alone.

Inference already ran on this hardware. Cached-score post-processing should require much less compute than repeated model inference, but its measured cost remains to be reported. For final timing, warm up, synchronize, measure repeated forwards, and state timer boundaries. Use the same memory terminology consistently.

## 9. Schedule and fallback

| Course weeks | Planned output |
| --- | --- |
| 7 | Audited baseline evidence, failure case, proposal presentation |
| 8-9 | Reproduction manifest, cached scores, development calibration |
| 10-11 | Frozen partial-cloud comparison, uncertainty intervals |
| 12-13 | Full-cloud control, per-affordance analysis, qualitative failures |
| 14-15 | Reproducible report and final presentation |

The main risk is that decreasing `none` merely replaces missed regions with incorrect foreground labels. The fallback is an honest negative result demonstrating the limits of the decision rule. No large-scale retraining or additional GPU acquisition is required for the primary study.

## 10. Items to complete before submission

- Add the team number and member names to the title slide. The assignment describes a team submission even though this draft supports individual preparation.
- Export the final deck with the required name `teamNN_T14_proposal.pdf`, after replacing NN with the actual team number.
- Recover the E01 full-shape checkpoint hash, exact data scope, and run settings if those historical rows remain in the deck. The fully documented E05 partial row is the strongest evidence.
- Package the actual local source diff and an exact object/view manifest. The upstream commit by itself does not reproduce the adapted run.
- Verify and disclose the requested seen/unseen grouping. Do not invent group scores from replacement phrases.
- Use the honest MPS memory label. If the instructor requires an exact physical peak value, explain the counter limitation and seek an accepted measurement convention.
- Rehearse the 10 main slides to 8 minutes. Keep the four appendix slides for Q&A.
- Submit the slide PDF on e-Class **before October 15, 2026, 10:59 AM KST**. The guideline rejects late decks. The talk is 8 minutes in English with 2 minutes of live Q&A.

## Local evidence sources

The deck's speaker notes cite the relevant evidence files. Main records:

- `research_archive/2026-10-05/E01_openad_original_baselines/` for historical full/partial logs.
- `research_archive/2026-10-05/E05_openad_dataset_analysis/summary.json` for the all-view MPS baseline.
- `research_archive/2026-10-05/E04_openad_mps_check/sample_regions.csv` for the vase point-count diagnostic.
- `research_archive/2026-10-05/E02_openad_selected_failures/figures/failure_pour.png` for the observed qualitative failure.
- `research_archive/2026-10-11/openad_sample452_mps_profile.json` for measured cost.
- `research_archive/2026-10-05/code/openad/config/openad_pn2/partial_view_open_vocab_cfg.py` for query mapping.
- `research_archive/2026-10-05/code/openad/utils/eval.py` for the metric convention.

The proposal follows the provided Week 7 guideline. The proposed hypothesis, bias grid, 5 pp guardrail, and development protocol are new design choices, not instructions quoted from that document or completed experiment results.
