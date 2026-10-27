# Three-method exploratory pilot

Prepared 2026-10-04. No Espresso or Affostruction inference has run.

## Ready

- Official Affostruction source pinned at bd37f11ea4e8f0ce211b46559a82e8fcc1e6609d (sparse checkout; excludes large assets).
- Unofficial Espresso-3D source pinned at b33d608d10e1619984cfc90d7b66fb218f9b56bb, inspected only.
- `pilot_manifest.json`: ten fixed Affogato objects, selected from the official 1,000-object Affostruction test manifest by a deterministic hash ordering, query index 0.
- `prepare_manifest.py`: regenerates the selection and verifies unique source IDs.
- Existing OpenAD baseline remains a separate completed experiment.

The official Affogato GitHub endpoint was checked directly and returned repository-not-found. This does not rule out private access. The unofficial implementation is not a reproduction of the exact official system.

## Execution blockers

Affostruction's dependency file restricts its environment to Linux x86_64 and includes CUDA-specific packages such as spconv-cu124 and CUDA rasterizers. Espresso's inspected inference and encoder code explicitly allocate CUDA tensors and use CUDA autocast. No validated MPS port was found or attempted; this is not proof that porting is impossible. Minimum inference VRAM is not established by this review. Do not assume training GPU memory is the inference minimum.

The user currently has only an M3 Pro. Obtain compatible compute before installing the supplied stacks. Use separate environments; preserve the working OpenAD environment. No weights or large datasets were downloaded.

## Data gate

For each fixed UID obtain the annotated XYZ/heatmap array, exact query text, original geometry, and RGB-D observations with camera metadata. The official processing instructions describe separate annotation, render and mesh inputs. Missing assets remain recorded as missing; do not silently replace difficult objects.

Inspect annotation shapes, coordinate transforms, units, point order, image/depth alignment and query correspondence. Audit the exact Espresso checkpoint's training objects against the selected UIDs. A test split for Affostruction is not automatically a test split for Espresso. Until overlap is resolved, results are exploratory, not generalization evidence.

## Experiment A: localization with supplied complete geometry

Use the same object/query pairs and a shared annotation evaluation surface. Run OpenAD, Espresso-3D (explicitly labeled unofficial unless official access becomes available), and Affostruction's affordance stage with complete geometry. Preserve each model's required preprocessing and record point counts/resampling. Retain transforms and map outputs back to the common surface using a validated mapping; record mapping distances and unsupported points.

Use the pinned official point-level metric definitions for aIoU, AUC, SIM and MAE. Inspect degenerate cases and report exclusions. Affostruction's metric API expects logits: avoid applying sigmoid twice to exported probabilities. Do not mix these scores with the existing multiclass mIoU.

OpenAD adapter is an unresolved design item, not implemented: a one-query softmax always equals 1. Audit the paper's OpenAD grounding baseline before choosing a scoring rule. If a fixed query-plus-background vocabulary is used as a fallback, label it an adaptation, freeze the rule in advance and document its calibration limitations. Do not tune it on this pilot's labels merely to improve the ranking.

## Experiment B: partial observation

Start from the same camera view. OpenAD and Espresso receive visible point geometry; Affostruction receives the corresponding RGB-D/camera inputs and reconstructs geometry. This compares end-to-end systems with different input modalities; it does not isolate reconstruction's causal benefit.

Evaluate visible-region localization separately from full-surface recovery. Do not fill hidden surfaces using nearest visible-point scores and silently call them observations. Define and validate visibility masks and missing-surface handling before reporting metrics. Affostruction's native voxel partial protocol remains a separate table unless all outputs are explicitly aligned to its grid and scoring rules.

## Outputs and interpretation

For every run save object UID, query, model source and commit, checkpoint hash, seed, input condition, preprocessing, raw scores, output geometry/point mapping, metric scope, runtime and memory. Save failures as well as successes. Produce one qualitative panel per object and a per-object table before aggregate summaries.

This ten-object set is development evidence for choosing a hypothesis, not a statistically strong benchmark or final held-out test. The later hypothesis test needs a separate object set and a one-factor ablation. A better model score alone does not identify which architecture, training data or geometry difference caused it.

OpenAD's `none` diagnostic is model-specific: the newer query-conditioned heatmap models do not necessarily have an equivalent background competitor. Compare localization quality across models, not an invented common `none` rate.

## Concrete request for the professor (draft, not sent)

I have completed OpenAD baseline and dataset-wide failure analysis. I prepared a ten-object Affogato pilot to compare OpenAD, Espresso-3D and Affostruction. The newer available implementations require CUDA, while I have only an M3 Pro. Could I access a Linux/NVIDIA machine for pretrained inference? The official Affogato code endpoint is inaccessible; do you have official Espresso-3D access, or may I use an explicitly labeled unofficial implementation? If compute is unavailable, may I retain a reproducible OpenAD experiment and document the newer-method execution constraints?

## Sources

- https://github.com/chrockey/Affostruction
- https://github.com/chrockey/Affostruction/blob/main/DATASET.md
- https://github.com/chrockey/Affostruction/blob/main/pyproject.toml
- https://junha-l.github.io/affogato/
- https://github.com/LHC0312/espresso-3d-reimplementation
