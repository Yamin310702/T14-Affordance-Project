# T14 proposal baseline audit

Audited 2026-10-11 from the archived OpenAD run records and result logs. This file separates measured results from missing proposal metadata; it does not claim any new inference was run during this audit.

## Baseline results already measured

| Run | Input condition | Data/evaluation scope | mIoU | Other recorded result | Interpretation |
|---|---|---|---:|---:|---|
| E01 | Full point cloud | Original OpenAD full-shape baseline | 14.4121% | Point accuracy 46.3089%; mean class accuracy 19.5370% | Historical released-method baseline log. |
| E01 | Partial point cloud | Original OpenAD partial-view baseline | 12.4964% | Point accuracy 45.2476%; mean class accuracy 17.3579% | Historical released-method baseline log. |
| E05 | Partial point cloud | 9,140 views; MPS; batch 1; point-weighted over views | 12.5364% | 2,416.95 s elapsed; repeated views/objects are not independent | Later full-dataset run. Do not present as numerically identical to E01: the saved record says stochastic point sampling and batch size differ. |

The full and partial E01 values should be the headline baseline table, with the E05 result clearly labeled as a later, all-view partial analysis. Keep both settings separate.

## One-input run and observed failure

OpenAD PointNet++ on partial validation sample 452 was run on CPU and MPS with the same checkpoint and seed. Both recorded mIoU 0.5962% for that one sample. The historical summaries reported 0.335 s on CPU and 0.874 s on MPS, but did not define timer boundaries. A new, instrumented MPS profile then measured one synchronized forward in **0.546 s after model and data setup**; it was a single cold forward without warm-up, so it is not a runtime benchmark.

The saved diagnostic for this same Vase sample shows an actual failure on **pour**: there were 563 ground-truth target points; all 563 were assigned the `none` argmax class. In the failure-analysis output, mean predicted pour score on those target points was 0.0584, while `none` averaged 0.6440. The new profiled forward again predicts `none` for the target region (pour IoU 0). The precise probabilities vary across saved single-sample evaluations, so use one run's scores and its matching visualization together in the slides.

## Reproduction details present in the saved run records

- Model: OpenAD PointNet++ (`OpenAD_PN2`), released pretrained checkpoint.
- Checkpoint SHA-256: `cbcfe9ff70448d97a7047f1ade817645a9be29dd2782fba57776792a8210f2ae`.
- Source commit recorded for E03–E05: `b082265ed085455a3fdc80b0d8d95dc51351e7b4`.
- Device: Apple M3 Pro using PyTorch MPS for E04/E05; the M3 Pro has unified CPU/GPU memory.
- Profiled partial sample 452 (2,048 points; shape `edb8e199485be4e968747a7f12307bdb`): 0.5456 s synchronized forward, excluding model/data setup; 5 ms sampled high-water Metal-driver allocation 1,390,460,928 bytes (about 1.295 GiB total process allocation); after-forward tensor allocation 727,330,816 bytes. These are sampled/current allocator statistics, not a guaranteed hardware peak or a separate VRAM measurement.
- E05: seed 1, batch size 1, 9,140/9,140 partial views, 40.28 minutes elapsed.
- Evaluation code and configuration were locally adapted; describe the baseline as the **public OpenAD checkpoint with an MPS-adapted evaluation** and list changes from the released setup. Do not imply the original CUDA environment was used unchanged.

## Items to complete before the proposal slides

1. **Verify the official seen/unseen grouping.** The saved E01 logs contain per-class IoUs, but neither E01 nor E05 summary records a verified seen/unseen aggregate. The OpenAD config uses separate `train_affordance` and `val_affordance` query lists; do not label the rows “seen” or “unseen” until the professor's exact grouping is mapped to these queries. Then calculate macro mIoU per group separately for full and partial settings from the archived per-class values, if the mapping supports it.
2. **Report memory with the right label.** The profiler recorded a 5 ms sampled high-water of Metal-driver allocation; this is a useful measured estimate, not a guaranteed physical peak. Apple Silicon uses unified memory rather than a separate VRAM pool, and PyTorch's MPS functions do not expose the CUDA-style peak counter. The report gives both total driver allocation and tensor allocation so they are not confused.
3. **Keep the timing scope visible.** Use the 0.5456 s synchronized forward time, explicitly labeled as a single cold forward after setup with no warm-up. Do not call it a benchmark or compare it as steady-state latency.
4. **Freeze the proposed test protocol.** Use one modification only, such as k-NN smoothing of OpenAD point scores on partial clouds. Compare against the unchanged checkpoint on identical inputs and the same official point-IoU/mIoU metric. Keep full-cloud performance as a separate control. Choose all smoothing parameters on a development split, never on the held-out test split.

## Proposal hypothesis candidate

**Research question:** Does local k-NN smoothing of OpenAD's point scores improve affordance localization on partial point clouds, especially for unseen affordance queries?

**Falsifiable hypothesis:** On the locked evaluation split, k-NN smoothing will increase macro point-wise IoU for the predefined unseen-query group on partial clouds relative to unchanged OpenAD. If the held-out unseen-group IoU does not increase, the hypothesis is not supported. Report seen-query and full-cloud results separately to expose regressions.

This is a proposed experiment, not a result. Affogato and Affostruction may be introduced as related work or later comparison methods; no results from them belong in the OpenAD baseline table.

## Source records

- `research_archive/2026-10-05/E01_openad_original_baselines/full_shape_baseline_result.log`
- `research_archive/2026-10-05/E01_openad_original_baselines/partial_view_baseline_result.log`
- `research_archive/2026-10-05/E03_openad_cpu_check/summary.json`
- `research_archive/2026-10-05/E04_openad_mps_check/summary.json`
- `research_archive/2026-10-05/E05_openad_dataset_analysis/summary.json`
- `research_archive/2026-10-05/E05_openad_dataset_analysis/per_affordance.csv`
- `research_archive/2026-10-05/E02_openad_selected_failures/user_reported_progress.txt`
- [Slide-ready baseline values](OPENAD_PROPOSAL_BASELINE_TABLE.csv)
