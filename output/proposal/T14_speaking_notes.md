# T14 speaking notes and Q&A

Ten main slides, planned duration 8:00. Four appendix slides follow. Rehearse aloud and shorten any section that runs over its target. Add team details before submission.

## Slide 1: Background suppression in 3D affordance prediction (15 seconds)

My proposal studies a failure in a pretrained 3D affordance model. I will test whether one background-score adjustment can recover missed regions on partial point clouds.

## Slide 2: Task and research scope (60 seconds)

The input is an object point cloud and an affordance text query, such as pour. The output is a score or label at each observed point. This tells us which surface region supports an action. In our archived OpenAD implementation, inference scores a fixed query list jointly and predicts the highest-scoring label at each point. I focus on partial clouds because a depth sensor observes only part of an object. My study evaluates the visible points. It cannot recover missing geometry or establish physical robot success.

## Slide 3: Baseline choice and setup (60 seconds)

I chose OpenAD because the topic guide recommends it and its official repository supplies code and pretrained weights. The baseline status is public. I use the released PointNet++ checkpoint with local MPS adaptations on an Apple M3 Pro. The point-text model stays frozen. My primary evidence comes from a completed evaluation of all 9,140 partial validation views. There are 18 affordance queries plus none. The archived evaluator pools point counts across views and then averages class IoUs. Local device and batch-size changes mean I report an adapted evaluation of the public checkpoint, rather than an unchanged CUDA reproduction.

## Slide 4: Measured baseline evidence (55 seconds)

These are measured results from the archive. The historical full-shape run obtained 14.4121 percent mean IoU, and the historical partial run obtained 12.4964 percent. A later all-view MPS partial evaluation obtained 12.5364 percent on 9,140 views. I keep these records separate because the source notes different stochastic sampling and batch sizes. These numbers are not a controlled estimate of the effect of occlusion, nor a comparison with Affogato or Affostruction. For the proposed method comparison, I will evaluate both methods on exactly the same saved inputs with the same class-IoU calculation.

## Slide 5: Measured cost on one input (50 seconds)

A saved instrumented run used partial validation sample 452, a vase with 2,048 points, on the M3 Pro integrated GPU. The synchronized forward took 0.5456 seconds after model and data setup. This was one cold forward without warm-up, so it is not steady-state latency. Sampling the Metal driver allocation every five milliseconds observed a high-water value of approximately 1.295 gibibytes. Apple Silicon uses unified memory. This is an allocation estimate, not a guaranteed physical peak VRAM value. The current tensor allocation after the forward was about 0.677 gibibytes. I will repeat profiling with warm-up for the final experiment.

## Slide 6: Observed failure: pour becomes none (75 seconds)

This figure comes from our archived run, not from a paper. The orange region contains annotated pour points, while the prediction contains no pour region. In the matching saved failure diagnostic for this vase, all 563 pour points received none. Pour IoU was zero. This observation supports a possible explanation: the background class may dominate the decision even when the point features retain some useful affordance information. That is only a hypothesis. If the underlying features do not rank the correct foreground action, lowering the background score will simply produce wrong labels. The experiment will distinguish these explanations.

## Slide 7: Research question and hypothesis (45 seconds)

My question is whether adjusting only the none score improves partial-cloud localization without excessive false positives. I propose a single scalar bias, selected using development objects and then frozen. The hypothesis is that this adjustment will increase mean IoU across the same 19 classes relative to unchanged OpenAD, while increasing the background-point false-positive rate by no more than five percentage points. I will support the hypothesis only if the paired object-bootstrap confidence interval for the IoU gain lies above zero and the false-positive guardrail holds. An IoU loss, an uncertain gain, or too many false positives would leave the hypothesis unsupported. Five percentage points is a predeclared study tolerance, not a robot safety limit.

## Slide 8: One change, fixed comparison (75 seconds)

The output module is the only change. For each point, I subtract one scalar b from the none log score, keep the other scores unchanged, and take argmax. At b equals zero this is exactly the baseline. I will cache scores and use the same points, checkpoint, query ordering, and metric for every comparison. On development objects from the official training split, I will choose b from zero, point two five, point five, one, one point five, two, and three. I will maximize the official 19-class mean IoU subject to the false-positive constraint, breaking ties toward the smaller bias. I then freeze the selected bias and reevaluate the official validation split. Existing validation inspection means this is exploratory evaluation. Ablations include the bias curve and a full-cloud control. I will group all views of an object together and bootstrap objects, not individual views.

## Slide 9: Milestones through Week 15 (30 seconds)

By Week 7 I will finish the reproduction manifest and presentation. Weeks 8 and 9 cover cached scores and development calibration. Weeks 10 and 11 cover the frozen comparison and uncertainty estimates. Weeks 12 and 13 cover full-cloud controls and failure analysis. Weeks 14 and 15 cover the final report and presentation. The main risk is that a background bias recovers points but mislabels them. The fallback is to report this negative result and the measured precision-recall trade-off. The project does not depend on obtaining additional GPUs or training a large model.

## Slide 10: Proposed contribution (15 seconds)

The contribution is a controlled diagnosis of background suppression in a frozen 3D affordance model. One scalar adjustment gives a feasible experiment on existing hardware. The result will tell us whether the observed failure comes partly from the decision rule or whether better features are needed.

## Q&A preparation

**Why this baseline?** OpenAD is the public candidate named in T14, with a working pretrained checkpoint on the intended dataset. A single output adjustment fits the available hardware and isolates an interpretable mechanism.

**What is new?** The contribution is a controlled diagnosis of background suppression, including a locked bias and explicit false-positive costs. We do not claim that scalar score calibration itself is a new general algorithm.

**Why not geometric smoothing?** The observed failure assigns the target region entirely to none. Neighbor smoothing may preserve that decision. A background bias directly tests the suspected decision competition, although the hypothesis can fail.

**Could lowering none produce arbitrary foreground predictions?** Yes. The foreground ranking stays unchanged. That risk motivates per-class IoU, background error measurement, and a predeclared guardrail.

**Why 5 percentage points?** It is a proposed tolerance for the study, selected before calibration and evaluation. It is not a physical safety standard. Report the whole trade-off so readers can judge other tolerances without choosing one from the validation outcome.

**How do you avoid leakage?** Verify object-ID separation, calibrate only on development objects, and group all views of each object. Since validation has already been examined, the proposal calls the reevaluation exploratory and does not claim an untouched test.

**Do new query words mean new action concepts?** No. The archived replacement list contains corresponding phrases such as pourable and pour. A verified grouping is needed before claiming seen/unseen action results.

**Does the 1.295 GiB value meet the peak VRAM requirement?** It reports an observed allocation high-water on Apple unified memory, sampled every 5 ms. It does not prove an exact physical peak. Explain the measurement limitation and record the counter used.

**Are full and partial scores directly comparable?** They share a class-IoU convention, but input conditions, checkpoints, and archived sampling can differ. Compare adjusted versus unchanged predictions within each setting. The full/partial difference is descriptive.

**What if the method fails?** Preserve the negative result and determine whether errors come from none dominance or wrong foreground ranking. That supports a concrete conclusion about which module would need improvement.

**Why not compare three models?** Existing Espresso tests use an unofficial reimplementation and a different protocol. Affostruction has not run. A fair multi-model comparison would require aligned inputs, query definitions, evaluation surfaces, splits, and compatible compute.

## Final rehearsal priorities

Spend the most time explaining the baseline evidence and the observed failure. State the question once, then explain the one changed scalar. Be ready to define the metric and false-positive rate without reading the slide. Keep the limitations concise and accurate.
