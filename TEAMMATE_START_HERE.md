# T14 research handoff: 3D affordance prediction

Prepared 2026-10-11 for sharing the project work and proposal context. This note is the recommended first read. The ZIP also contains the dated experiment archive, raw result tables, run records, figures, scripts, and the course proposal guideline.

## What this project is about

The course topic is **T14: 3D affordance prediction**. Given a 3D object point cloud and a natural-language action/query such as “grasp” or “pour,” the model assigns an affordance score or label to each point. The goal is to identify which parts of an object support an action, including when the observed point cloud is partial.

The proposal is motivated by two observed behaviors: OpenAD is sensitive to which text name is used for the same label slot, and a partial Vase example failed to identify its `pour` region. The work asks whether carefully fixed prompt wording can make per-point localization more reliable, while measuring false alarms on objects that do not have the target affordance.

## What has been done

1. **OpenAD baseline and failure analysis.** Public pretrained OpenAD PointNet++ checkpoint evaluated on the local 3D-AffordanceNet validation data. A concrete Vase `pour` failure has 563 target points, all assigned to `none` in the saved diagnostic. The checkpoint is frozen; this project has not retrained the full model.
2. **Seen-name versus held-out-query comparison (E17).** Both vocabularies were tested on the same 9,140 partial views and 2,285 full shapes. On partial views, 18-affordance macro mIoU was 40.39% for training-associated names and 10.77% for the held-out query names. On full shapes it was 35.18% and 7.87%. The complete CPU and Apple MPS runs closely match. This is evidence of query-vocabulary sensitivity; the aligned names include synonyms, so it is not proof of generalization to new affordance concepts.
3. **Prompt wording studies with Espresso-3D and Affogato weights (E07–E16).** These were exploratory runs using a locally runnable, unofficial Espresso-3D reimplementation with Affogato-pretrained weights on selected OpenAD objects. Prompt effects varied by affordance. Positive localization gains sometimes came with more false alarms, and threshold calibration did not yield a useful universal operating point. These are not official Affogato benchmark results.
4. **Fixed prompt transfer to OpenAD (E18).** Five prompts selected before OpenAD evaluation were tested on 1,941 prior-study-disjoint validation shapes. The `unlock` paraphrase reached 12.36% global IoU, compared with 0.09% for its original sentence and 3.69% for the canonical word. It also increased negative-object false alarms (any-alarm rate 0.34% to 1.61%). The other four paraphrases did not show a comparable gain. CPU and MPS results reproduce the same conclusion.
5. **Affostruction.** A run preflight was documented, but Affostruction inference was not completed on this Mac. Do not cite an Affostruction metric, prediction, or reproduction as part of our results.

## What the proposal plans to test

**Research question:** On partial point clouds, does averaging several fixed, meaning-equivalent text prompts improve point-wise affordance localization compared with a single canonical query, and does the effect vary by affordance or increase false alarms?

**Hypothesis:** With OpenAD's point model and text encoder frozen, averaging a prewritten set of equivalent prompts will improve 18-affordance macro point-wise IoU over one canonical query on the predeclared evaluation objects. The effect will vary by affordance and may raise false alarms. The hypothesis can fail if the paired object-level improvement is non-positive or its 95% interval includes zero.

**Proposed comparison:** Keep the checkpoint, point clouds, labels, class slots, and prediction rule fixed. Compare prompt counts K=1, 3, and 5; K=5 is the primary condition. Use partial views as the primary setting and full shapes as a control. Report global IoU/mIoU, per-affordance results, AUC, and false-positive rates on negative objects. Freeze the prompts before examining evaluation labels/scores. Since earlier pilots used the available validation data, seek a fresh or object-disjoint evaluation set if available; otherwise clearly label follow-up results as exploratory validation.

This prompt-ensemble experiment is a **proposal, not a completed result**. See `research_archive/2026-10-11/T14_PROPOSAL_SLIDE_OUTLINE.md` for the 10-slide, 8-minute talk plan and resource/reporting details.

## How to read the package

- `research_archive/2026-10-11/README.md` is the index for the current archive.
- `research_archive/2026-10-11/ALL_EXPERIMENTS_PROPOSAL_MAP.md` separates completed, exploratory, aborted, teammate-generated, and unrun work. It explains which metrics must not be compared directly.
- `research_archive/2026-10-11/PROPOSAL_BASELINE_AUDIT.md` and `OPENAD_PROPOSAL_BASELINE_TABLE.csv` contain the OpenAD baseline and measurement caveats.
- `research_archive/2026-10-11/seen_query_comparison/` and `seen_query_comparison_mps/` contain E17 CPU/MPS records and per-class/per-view tables.
- `research_archive/2026-10-11/E18_openad_prompt_transfer/` contains E18 CPU/MPS protocol, summary and per-object results.
- `research_archive/2026-10-11/E11...E16*/` and `2026-10-08/E08...E10*/` contain the exploratory Espresso/Affogato-weight studies and their supporting data.
- `research_archive/2026-10-11/AFFOSTRUCTION_PREFLIGHT.md` records why no Affostruction result is claimed.
- `research_archive/2026-10-11/proposal_guideline.pdf` is the professor's assignment guideline. Its requirements are instructions for the proposal, not experimental findings.

## Reproduction and limits

Run metadata records the checkpoint, data scope, device, and script details. The ZIP does **not** include the complete OpenAD repository, model checkpoints, or full 3D-AffordanceNet data. Those must be obtained separately and used in accordance with their licenses and instructions. The saved scripts expect the local source/data paths written in their run records and may need path changes on another computer.

The proposal presentation guideline gives the Week 7 submission deadline as **10:59 AM KST, October 15, 2026**. The submitted PDF naming format is `teamNN_T14_proposal.pdf`. Confirm team number/member details before producing the final deck.
