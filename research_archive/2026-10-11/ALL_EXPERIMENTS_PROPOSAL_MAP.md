# T14 experiments: complete evidence map for the proposal

This inventory includes every experiment currently preserved in the research archive, plus the teammate's qualitative artifact and the not-yet-run Affostruction work. It distinguishes measured results from exploratory evidence and planned work. A result from one model or metric must not be presented as if it came from another.

## What belongs in the proposal

The 8-minute proposal should show the complete research path, but not every row in the main narrative. Put the required OpenAD baseline and its actual failure in the main slides. Briefly report the later prompt experiments as preliminary evidence that motivated the research question; put the detailed action-by-action tables, threshold studies, device checks, and provenance notes in backup slides or this appendix. This uses all the work without implying that incomparable numbers are a single benchmark.

The current proposal hypothesis in the baseline audit is k-NN smoothing on OpenAD. It remains a *planned* OpenAD experiment, not a finding. E08–E16 give useful exploratory motivation for studying localization versus false alarms and prompt wording, but those experiments use an unofficial Espresso-3D reimplementation with Affogato-pretrained weights on OpenAD partial views. They do not directly validate an OpenAD k-NN hypothesis. The final hypothesis should be chosen to match the professor-approved proposal scope and the code/results actually available.

## A. OpenAD baseline and failure evidence

| Run | Status and setup | Result to retain | What it can support / caveat |
|---|---|---|---|
| E01 | Completed; public OpenAD pretrained checkpoint; full and partial point clouds | Full: mIoU 14.4121%, point accuracy 46.3089%, mean class accuracy 19.5370%. Partial: mIoU 12.4964%, point accuracy 45.2476%, mean class accuracy 17.3579%. | Required baseline anchor. Preserve full and partial separately. The recorded setup is an adapted evaluation; do not call it an untouched official reproduction. |
| E02 | Completed selected-failure inspection | Poor examples included clothe/wrap/pour; original stdout is incomplete. | Qualitative motivation only. These were selected failures, not a random or representative sample. |
| E03 | Completed CPU check, one partial object (sample 452) | Sample mIoU 0.5962%; historical reported time 0.335 s. | CPU feasibility check only. Timing boundaries were not recorded, so do not compare as a benchmark. |
| E04 | Completed MPS check, same one-object case | Sample mIoU 0.5962%; historical reported time 0.874 s. | Supports that inference works on M3 Pro MPS. Small CPU/MPS score differences can occur; this one sample is not a speed comparison. |
| E05 | Completed MPS analysis, 9,140 partial views, batch 1 | Point-weighted mIoU 12.5364%; 2,416.95 s elapsed; 9,140/9,140 views. | Broad partial-view result. Views/objects are repeated and not independent. Do not pool with E01: sampling/batch conditions differ. |
| E03/E04 detailed sample-452 diagnosis | Completed, OpenAD PN2, Vase, target “pour” | In the saved user diagnosis, 563/563 target points had `none` as argmax; mean `none` score 0.6440 vs `pour` 0.0584. Target-region pour IoU was 0. | Strong concrete failure example. Pair this score report with its own matching visualization; scores vary slightly across one-sample runs. |
| MPS profile, sample 452 | Completed one synchronized forward after setup; 2,048 points | 0.5456 s for one cold forward, no warm-up. Sampled high-water Metal allocation about 1.295 GiB; current tensor allocation after forward about 0.677 GiB. | Report as a one-input resource profile, not steady-state latency or guaranteed physical peak VRAM. Apple Silicon has unified memory. See `openad_sample452_mps_profile.json`. |

The baseline detail and reproduction caveats are in [PROPOSAL_BASELINE_AUDIT.md](PROPOSAL_BASELINE_AUDIT.md) and [OPENAD_PROPOSAL_BASELINE_TABLE.csv](OPENAD_PROPOSAL_BASELINE_TABLE.csv).

## B. Espresso-3D / Affogato-pretrained prompt experiments

These are complete exploratory runs on OpenAD partial views using a locally runnable, unofficial Espresso-3D reimplementation and Affogato-pretrained weights. The prompt comparisons are paired within each run, but the samples, selected affordances, and metric settings differ from OpenAD's official baseline evaluation. Report these as *our exploratory cross-model experiments*, never as official Affogato benchmark scores.

| Run | Status / sample | Main result | Proposal interpretation and limit |
|---|---|---|---|
| E06 | Completed compatibility test; CPU text encoder and MPS encoder/decoder; random weights/synthetic text | Tensor/device path ran. | Engineering smoke test only; no semantic prediction or accuracy result. |
| E07 | Completed single-case inference; Affogato-pretrained weights in unofficial Espresso-3D; OpenAD partial Vase, “pour” | Finite score maps; 1.643 s single reported inference after loading; 1.795 GB current allocation after inference. | Confirms this local path can produce outputs. One case, not a benchmark; current allocation is not peak memory. |
| E08 | Completed, one selected Vase, three paraphrases | One phrase improved aIoU 16.42%→18.20%; another phrase and the ensemble were worse (ensemble 12.45%). | Prompt wording can matter, but one selected object cannot establish an effect. |
| E09 | Completed, 10 selected positive “pour” objects | Original aIoU 6.10%; phrase 1 6.84% (paired interval +0.36 to +1.20 pp); other phrases/ensemble lower. | Small development result; objects were selected positives. |
| E10 | Completed, 20 held-out positive “pour” objects | Original 9.47%; phrase 1 9.99% (delta +0.52 pp, interval −0.31 to +1.35); phrase 3 and ensemble were worse. | Phrase-1 gain is uncertain; not a negative-object or prevalence test. |
| E11 | Completed, 20 fresh positive “grab” objects | Original 12.89%; “grasp” phrase 16.31% (delta +3.43 pp, interval −1.67 to +10.06); other prompt variants mixed. | Suggestive but imprecise and selected-positive only. |
| E12 | Completed CPU study, 20 fresh positive objects for each of five affordances; same 100 cases rerun on MPS | Original→selected phrase aIoU: grab 11.13→11.24%; pour 11.92→12.84%; wrap 6.83→6.94%; accommodate 2.09→3.61%; unlock 3.76→4.66%. Averaging prompt variants often lowered aIoU; phrase effects varied by action. CPU/MPS metrics matched to displayed precision. | Best evidence that effects are action-dependent and prompt averaging is not reliably beneficial. Still positive-only and from the unofficial Espresso/Affogato-weight path. |
| E13 | Completed MPS confirmation; 40 fresh positive objects per affordance (200 total) | Selected phrase delta in aIoU: grab +5.85 pp (CI +1.59,+11.00); pour +0.44 (−0.12,+0.99); wrap −0.73 (−1.35,−0.19); accommodate +0.34 (−0.19,+0.93); unlock +0.56 (−0.30,+1.65). | Supports a possible affordance-specific prompt effect, especially for grab, while showing no general benefit. Positive-only. |
| E14 | Completed MPS mixed-presence study; 20 positive + 20 negative per action | Positive localization deltas: grab +1.56 pp (CI −1.94,+5.74), pour +0.47 (−0.03,+0.94), wrap +0.41 (−0.48,+1.22), accommodate +0.04 (−0.49,+0.69), unlock −0.73 (−2.10,+0.43). At diagnostic threshold .5, accommodate negative alarms rose 6/20→13/20 for the selected phrase. | Central caution: positive-only gains do not ensure useful mixed-presence behavior. Stratified sample rates are not population prevalence estimates. |
| E15 | Completed threshold-calibration study; thresholds fit on 40 positive + 40 negative training cases and applied to E14 validation | Balanced-IoU objective selected cutoffs that collapsed to all-background/empty predictions; original prompt detected 0/20 positives for every action. | Important negative result: the chosen objective/threshold procedure was unusable. Do not report it as a successful method. Explain the failure and keep it as a lesson. |
| E16 | Completed “accommodate” operating-point study; 40+40 calibration and fresh 20-positive/20-negative validation | Predeclared ≤10% negative-alarm cap selected threshold 1.00 for both prompts, yielding zero positive detections. Threshold-free aIoU 3.64→5.99% (delta +2.35 pp, paired CI +0.26 to +5.33); AUC 38.4→42.4. At diagnostic .5, positives flagged 1/20→8/20, but negative alarms rose 4/20→19/20. | Strong example of localization-versus-false-alarm tradeoff, not evidence of a deployable operating point or universal prompt improvement. |
| E12/E16 MPS checks | Completed same-input CPU–MPS comparisons | E12 metrics match at displayed precision. E16 score maps differ by at most 6.29×10⁻⁶ over 400 arrays; thresholds and conclusions match. | Good reproducibility/device validation. Does not add independent examples or statistical power. |
| E16 five-action CPU attempt | Aborted and archived | No complete result; CPU scoring was impractical and MPS unavailable for that attempt. | Count it in the research record as aborted, never as an experiment result. |

Detailed run-level files are linked from [README.md](README.md); use the individual `RESULTS.md`, `DEVICE_COMPARISON.md`, or `NOTE.md` files for protocol and raw tables.

## C. Teammate's Affogato repository artifact and Affostruction

| Item | Status | What it is / how to use it |
|---|---|---|
| Teammate monitor point-cloud screenshot and `look.py` preview | Exploratory, teammate-generated, not yet independently verified | The screenshot reports a 16,384×8 `xyzc` array and a Monitor object's `queries.json`, then plots channels 3–7 as heatmaps. The code assumes those five columns are affordance heatmaps. Until the repository's saved-array schema and query/channel ordering are checked, call it a qualitative visualization artifact—not verified model inference or a metric result. Keep the script, source object, query file, checkpoint, and exact commit together if it is to be cited. |
| Affogato itself | No verified quantitative run in this archive | The E07–E16 local pathway uses Affogato-pretrained weights inside an unofficial Espresso-3D reimplementation; that is not a reproduction of Affogato's full official evaluation. |
| Affostruction | Preflight checked; inference not run | There are no Affostruction metrics or predictions. The CLI fails before parsing because `rootutils` is missing; beyond that, the released environment requires Linux x86_64/CUDA 12.4 and CUDA extensions, while the available runtime is macOS arm64 without CUDA and has none of the required extensions, checkpoint, or processed Affogato input. See [AFFOSTRUCTION_PREFLIGHT.md](AFFOSTRUCTION_PREFLIGHT.md). Mention official results as related-work context, not as our result. |

## Proposal-ready interpretation

The complete evidence supports these cautious claims:

1. The OpenAD pretrained baseline runs on the available Mac with an MPS-adapted evaluation, and the archived full/partial results provide a baseline anchor.
2. A concrete OpenAD failure is a partial Vase where the “pour” target region is classified as `none` throughout. This motivates testing how to improve point-level localization under partial observation.
3. Exploratory Espresso-3D/Affogato-weight prompt tests found that prompt changes can affect aIoU, but the effect is action-specific, prompt averaging is not consistently helpful, and better positive localization can sharply increase negative-object alarms.
4. Therefore no general “best prompt” or successful threshold is established. Affostruction remains untested.

Keep metrics separate: OpenAD E01/E05 report mIoU and point accuracy under their OpenAD evaluation; E08–E16 report threshold-averaged IoU (aIoU), AUC, or alarm counts for different experimental subsets and a different model path. Do not compare their percentages as though they shared a benchmark protocol.

For the talk, a defensible storyline is OpenAD baseline → observed partial-cloud failure → concise exploratory evidence from E12–E16 showing the prompt/localization tradeoff → one falsifiable, professor-approved next experiment. Detailed numeric results from every run belong in the appendix/evidence map, not necessarily the 8-minute main deck.

## Files to cite in slides or appendix

- OpenAD baseline: [baseline audit](PROPOSAL_BASELINE_AUDIT.md), [CSV table](OPENAD_PROPOSAL_BASELINE_TABLE.csv), and the source records listed there.
- Prompt study: E08–E16 detailed results linked in [README.md](README.md).
- MPS profile: [profile JSON](openad_sample452_mps_profile.json) and [profile script](profile_openad_single_input.py).
- Proposal requirements: the professor's [proposal guideline PDF](/Users/yaminmyataung/Downloads/proposal_guideline.pdf).
