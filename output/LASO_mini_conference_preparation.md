# Team 07: LASO mini-conference preparation

Prepared October 27, 2026. Working outline for the October 29 session, assuming LASO is reserved for Team 07. This file does not confirm a reservation or submission.

## Presentation focus

**Paper:** LASO: Language-guided Affordance Segmentation on 3D Object, CVPR 2024.

**Central question for the audience:** How can a natural-language question identify the actionable region of a 3D object?

**Connection to our project:** We investigate whether fixed paraphrases and averaging three score maps improve frozen OpenAD localization without increasing false alarms. LASO supplies related task design and a learned alternative. Its results do not establish our proposed improvement.

## Verified paper facts

The following short factual digest draws on the publisher abstract [1] and paper [2].

- LASO names the task and dataset. PointRefer names the proposed model.
- Input: object point cloud and an affordance question. Output: the relevant region.
- Dataset: 19,751 point–question pairs, 8,434 shapes, 870 questions, 23 object categories and 17 affordance types.
- PointRefer combines language and point features at multiple scales. Question-conditioned queries produce dynamic kernels for mask prediction.
- In the unseen setting, selected object-category/affordance combinations are withheld from training. The seen and unseen settings share validation and test sets.
- Reported PointRefer mIoU: 20.8% seen and 14.6% unseen. IAGNet: 17.8% and 12.9%. These are LASO benchmark values, not our OpenAD measurements.
- Ablation mIoU: full model 20.8%, without adaptive fusion 19.8%, without the referred point decoder 18.7%.

## Proposed 12-minute talk

The titles, timing, teaching examples and discussion prompts below are our presentation plan. They are not additional experimental findings.

| Slide | Time | Purpose and speaking guidance |
| --- | --- | --- |
| 1. LASO | 0:00–0:30 | Give the paper title, venue and Team 07 names. State the central question. |
| 2. Actionable regions | 0:30–1:30 | Use scissors as a teaching example: holding and cutting refer to different regions. Identify this as an illustrative example unless using an actual paper figure. |
| 3. The LASO task | 1:30–2:30 | Walk through the input and desired output. Ask the audience to predict the region before revealing an example. |
| 4. Dataset and evaluation | 2:30–3:45 | Use the factual digest. Explain the split with a hypothetical combination, clearly labeled as an illustration. |
| 5. PointRefer overview | 3:45–5:00 | Use the original Figure 4 with a visible citation. Trace the point and question branches. Explain how they meet before discussing individual modules. |
| 6. Language-conditioned prediction | 5:00–6:30 | Zoom into the fusion and decoding blocks in Figure 4. Explain what each block contributes. Keep equations in the appendix. |
| 7. Results | 6:30–7:45 | Show the selected benchmark comparison from the digest. Explain the metric and evaluation setting before reading numbers. Avoid comparing them numerically with our project. |
| 8. Component evidence | 7:45–8:45 | Show the two selected ablations. Explain that component removal supports a contribution within this trained system. |
| 9. What the evidence leaves open | 8:45–10:15 | Discuss the questions below. Distinguish our critical assessment from the authors' conclusions. |
| 10. Connection to Team 07 | 10:15–12:00 | State our three query conditions and the joint localization/false-alarm objective. Close with what this paper teaches us about designing a fair evaluation. |

## Suggested opening in English

“Suppose a robot sees a pair of scissors. Knowing that the object is scissors does not tell the robot which part to hold. The requested action determines the relevant region. Today we introduce LASO, a paper about connecting a question in natural language to a functional region on a 3D object.”

This is our teaching example, not a quotation from the paper.

## Critical discussion and project connection

These are our interpretations and follow-up questions, not claims that the authors demonstrated.

**Wording stability:** Would meaning-equivalent questions give consistent regions for an identical point sample? A benchmark score alone cannot answer that. Our controlled comparison changes the query while holding other factors fixed.

**Absent affordances:** What should the model return when the requested action is unsupported? We should evaluate verified negative object–query pairs separately. A low error averaged across points can conceal an occasional false region.

**Metric compatibility:** Before comparing two systems, audit input geometry, dataset split, thresholding, label definitions and aggregation. A shared metric name does not make two experiments comparable.

**Generalization:** Withholding an object–action combination and rephrasing an action test different abilities. Keep those distinctions visible during the talk.

**Project success criterion:** Predeclare the primary IoU comparison and the acceptable false-alarm change. Report a gain with an alarm increase as a trade-off. Do not promise that an improvement will occur.

## Q&A preparation

**Are we reproducing PointRefer?**
Our current experiment uses frozen OpenAD. Presenting a related paper does not require claiming a local reproduction.

**Does the paper validate prompt averaging?**
Our averaging experiment needs its own evidence. We use this paper to motivate language-conditioned localization and to discuss evaluation design.

**What if the ensemble helps some actions and hurts others?**
Report per-action changes alongside the aggregate. Keep the primary comparison fixed, and avoid choosing winners after inspecting the final evaluation.

**What if mean IoU improves but false alarms increase?**
Show both changes. The result may be useful, but it does not meet a strict joint-improvement objective.

## Preparation and submission

- Confirm the existing LASO reservation. The October 22 sign-up deadline has passed. If Team 07 is not registered, contact the instructor about availability.
- Use original, attributed paper figures when assembling slides. Check figure and table labels against the PDF before export.
- Rehearse the talk to 12 minutes, leaving the separate Q&A period available.
- Submit the slide PDF on e-Class before October 29, 2026, 10:59 AM KST, following the announcement. No submission has been made by this preparation step.

## Sources

[1] Publisher record and abstract: https://ieeexplore.ieee.org/document/10656061/

[2] Paper, especially Sections 3–5, Figure 4 and Tables 4–5: https://openaccess.thecvf.com/content/CVPR2024/papers/Li_LASO_Language-guided_Affordance_Segmentation_on_3D_Object_CVPR_2024_paper.pdf

[3] Author code and dataset repository: https://github.com/yl3800/LASO

The publisher abstract and indexed paper passages were accessible during preparation. Direct retrieval of the complete PDF returned HTTP 403, so this is a sourced preparation outline rather than a claim of a complete page-by-page paper review.
