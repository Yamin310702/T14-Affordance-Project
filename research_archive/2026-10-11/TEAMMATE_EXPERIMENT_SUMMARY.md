# T14 research experiment summary

Prepared 2026-10-11. This document is the compact record of the project from initial baseline through the current proposal. It distinguishes measured experiments from the proposed next experiment.

## Project question

**T14: 3D affordance prediction** asks a model to label each point of an object cloud with the action that point supports, given a text query. This project investigates OpenAD and how query wording affects point-level affordance localization, especially when the 3D observation is partial.

## Work completed

| Stage | Work and main result | How to interpret it |
|---|---|---|
| E01: OpenAD baseline | Released OpenAD checkpoint: full-input mIoU 14.4121%; partial-input mIoU 12.4964%. | Baseline anchor from the archived OpenAD evaluation. Report the adapted evaluation setup and keep full/partial settings separate. |
| E02–E05: failures and feasibility | M3 Pro/MPS inference works. In one partial Vase, 563 points labeled as `pour` were all predicted as `none`; the matching score report had mean `pour` 0.0584 and `none` 0.6440 on those target points. E05 evaluated 9,140 partial views and recorded 12.5364% point-weighted mIoU. | The Vase is a real observed failure, not a representative aggregate. The E05 protocol differs from E01; do not combine their values as an exact reproduction. A separate profile measured one cold 2,048-point forward at 0.5456 seconds. The ~1.295 GiB Metal value is a sampled unified-memory allocation estimate, not guaranteed peak VRAM. |
| E06–E07: Espresso setup | Compatibility smoke test, then one real single-object prediction using an unofficial Espresso-3D reimplementation with Affogato-pretrained weights. | Confirms that one local inference path can run. It is not an official Affogato experiment or benchmark. |
| E08–E11: early wording studies | Prompt wording changed aIoU on individual and selected positive objects. Examples: a `pour` phrase gain on 10 objects; the held-out 20-object interval included zero; a 20-object `grab` phrase gain also had a wide interval including zero. | Exploratory and small, often positive-only. These results motivated broader tests; they do not establish a general prompt gain. |
| E12–E13: multi-action confirmation | Five affordances tested on fresh positive objects. Effects varied by action; prompt averaging often lowered aIoU. E13 found a clearer gain for `grab`, while other actions were small or mixed. CPU/MPS E12 scores agreed to displayed precision. | Prompt effects are action-specific. Positive-only samples cannot measure false alarms on absent affordances. |
| E14–E16: mixed presence and thresholds | Added negative objects, tested thresholds, and quantified false alarms. Localization changes were mixed; a threshold calibration objective collapsed to empty predictions; a later ≤10% alarm constraint found no useful cutoff. For accommodation, a selected prompt improved threshold-free aIoU but sharply increased negative alarms at a diagnostic threshold. | Important negative and trade-off results. Do not claim a universal best prompt or a successful deployable threshold. The E16 five-action CPU attempt was aborted and is not a completed result. |
| E17: query vocabulary | Same frozen OpenAD checkpoint and inputs, comparing training-associated names with held-out query names. Partial 18-affordance mIoU: **40.39% vs 10.77%**; full: **35.18% vs 7.87%**. Complete MPS reruns match CPU within 0.006 percentage points on headline scores. | Strong query-vocabulary sensitivity, not proof of unseen-concept generalization. Several aligned names are semantic synonyms. |
| E18: prompt transfer to OpenAD | Five fixed paraphrases selected before OpenAD scoring; 1,941 prior-study-disjoint validation shapes. `unlock` paraphrase IoU **12.36%**, original sentence **0.09%**, canonical `unlock` **3.69%**. Negative-object any-alarm rate rose **0.34% to 1.61%** for selected vs original sentence. The other four prompts had no comparable gain. CPU/MPS reruns reproduce this conclusion. | Exploratory validation evidence for an affordance-specific wording effect with a false-alarm trade-off. Not an official held-out test result or general prompt rule. |

## Proposed research (not run yet)

**Research question:** On partial point clouds, does averaging several fixed, meaning-equivalent text prompts improve point-wise affordance localization over a single canonical query, and does the effect vary by affordance or increase false alarms?

**Falsifiable hypothesis:** With the OpenAD point model and CLIP text encoder frozen, a prewritten five-prompt ensemble will improve 18-affordance macro point-wise IoU over the single canonical query on predeclared evaluation objects. The effect may vary by affordance and may increase false alarms. The hypothesis is not supported if paired object-level improvement is non-positive or its 95% interval includes zero.

**Planned comparison:** K=1, 3, and 5 prompt embeddings, with K=5 as the predeclared primary condition. Keep checkpoint, inputs, labels, point samples, class slots, and prediction rule fixed. Partial views are primary; full shapes are a control. Report global and per-affordance IoU, AUC, and false-positive rates on negative objects. Freeze prompt wording before evaluation. Earlier pilots have used the available validation data, so a fresh or object-disjoint evaluation set is preferred; otherwise call the result exploratory validation.

## Boundaries and caveats

- OpenAD E01/E05/E17/E18 use the OpenAD checkpoint and report mIoU, IoU, point accuracy, or related OpenAD metrics.
- E07–E16 use Affogato-pretrained weights inside an unofficial Espresso-3D reimplementation and use different samples and metrics such as aIoU, AUC, and threshold alarms. Keep these numbers separate from OpenAD baseline numbers.
- No Affostruction inference was run. Affostruction belongs in related work or future work only.
- The teammate's Monitor screenshot and plotting script are described in the evidence map as qualitative and unverified; do not present them as an independent metric result.
- The package omits large raw prediction arrays and full datasets/checkpoints. The full archive ZIP remains available for raw intermediate arrays.

## Main proposal storyline

Task and input/output → public OpenAD baseline → observed partial Vase failure → query/prompt sensitivity from E17/E18 and exploratory Espresso/Affogato-weight studies → one controlled prompt-ensemble question → frozen-model protocol, ablations, false-alarm metrics, schedule, and risk.

Use “training-name” and “held-out-query vocabulary” rather than “seen/unseen affordances” unless the dataset split justifies that stronger wording. The professor's guideline asks for an 8-minute English proposal, an actual baseline and failure, one falsifiable question, a controlled plan, and reproducibility details. The slide outline and guideline PDF are included in the package.
