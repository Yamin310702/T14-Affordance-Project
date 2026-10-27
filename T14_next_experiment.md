# T14: feasibility review and next experiment

Checked 2026-10-04. Available hardware: Apple M3 Pro, 18 GB unified memory; no NVIDIA server access. This is a source review and proposed protocol, not a new experimental result.

## Verified availability

- Affogato introduces a dataset and the Espresso models; using its dataset is distinct from reproducing Espresso-3D. Its official project page links to `junha-l/affordance_segmentation`, which returned 404 during this review. Official runnable code and model weights were not verified. [Project](https://junha-l.github.io/affogato/)
- The public Affogato dataset repository lists 30.4 GB of archives. Dataset availability alone does not establish availability of all evaluation assets or split annotations. [Dataset files](https://huggingface.co/datasets/project-affogato/affogato/tree/main)
- Affostruction has official inference/evaluation code and pretrained checkpoints. Its documented environment is Linux x86_64 with CUDA 12.4. It evaluates affordance grounding on Affogato, with separate complete-geometry and reconstructed-geometry protocols. This provides a concrete connection between the recommended papers. [Official repository](https://github.com/chrockey/Affostruction)
- An unofficial Espresso-3D implementation and weights exist. Its usage specifies CUDA; it reports architectural and training departures from the paper. Do not label its outputs official Affogato reproduction. MPS compatibility and minimum inference memory remain unverified. [Implementation](https://github.com/LHC0312/espresso-3d-reimplementation) · [Weights](https://huggingface.co/LHC0312/espresso-3d-reimplementation)

## Decision

The official Affostruction environment cannot be followed directly on this Mac. A device-string change does not establish compatibility of native dependencies. Do not promise that both newer models can run locally. Do not discard the existing OpenAD baseline.

Clarify the required experimental scope with the professor using the draft below. If actual inference with both methods is required, request suitable university compute and official Espresso access, or agreement to use the explicitly labeled reimplementation. No messages have been sent.

## Proposed staged experiment

1. Run a supplied Affostruction example on compatible compute. Save input, query, heatmap, geometry, exact commit/checkpoints, seed, elapsed inference time, and peak device memory. A successful demo is a feasibility check, not a benchmark.
2. Prepare a fixed pilot of 10 objects from its supplied Affogato evaluation manifest. Verify object IDs, annotations, RGB-D/camera assets, and geometry alignment before inference. Expand only after the pilot works. Inspect the official data preparation instructions: https://github.com/chrockey/Affostruction/blob/main/DATASET.md
3. Run its complete-geometry and single-view reconstructed-geometry protocols. Use the repository's protocol-specific metrics; these settings are not interchangeable and their difference is not an isolated causal estimate of reconstruction benefit.
4. Add Espresso-3D after obtaining official access or accepting a labeled reimplementation. Audit training/test object overlap before treating a shared subset as held out. Use identical object-query pairs, coordinate conventions, evaluation surfaces and metric code wherever a direct comparison is claimed.
5. Keep the original OpenAD 3D-AffordanceNet results in a separate table. A new OpenAD evaluation on Affogato requires a documented query/scoring adapter and aligned evaluation protocol. Its existing multiclass mIoU must not be compared numerically with aIoU from another dataset.

Pilot success criterion: valid outputs and verified evaluation for every selected object, with one inspected failure. Method-improvement claims require subsequent controlled experiments.

## Work possible immediately on the Mac

- Preserve the OpenAD results and collect the missing reproduction metadata.
- Complete the dataset-wide target-versus-none diagnostic already motivated by the observed failures.
- Build the comparison protocol and inspect Affogato annotation/split formats before committing to a large download.
- Keep prompt ensembling provisional; it is not equivalent to implementing either recommended method.

## Draft question for the professor

I have reproduced OpenAD on my M3 Pro and analyzed several partial-view failures. I checked the two recommended papers: Affostruction has public code/checkpoints but requires Linux/CUDA, while Affogato's official code link currently returns 404. Its dataset is public, and an unofficial Espresso-3D implementation exists. Do you expect inference results from both newer models, or would an Affogato-data/Affostruction study with OpenAD experiments meet the scope? If both models must run, could I obtain university GPU access and use the unofficial Espresso implementation if official weights remain unavailable?

## Status

Completed: release and environment review, proposed pilot, professor clarification draft.
Not completed: new model installation, checkpoint download, inference, or benchmark reproduction.
