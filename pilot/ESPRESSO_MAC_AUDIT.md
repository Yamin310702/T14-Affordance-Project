# Espresso-3D Mac feasibility result

2026-10-05. Source: unofficial Espresso-3D reimplementation, commit b33d608d10e1619984cfc90d7b66fb218f9b56bb. This is an engineering compatibility experiment, not affordance accuracy evaluation or an official Affogato reproduction.

## Changes made

Added a construction-device argument in espresso/model.py, preserving CUDA as default. CUDA retains its existing bfloat16 autocast; non-CUDA uses float32 without autocast. Tensor shapes and network architecture are unchanged. Numerical equivalence with CUDA is untested. Patched source is confined to this workspace; original OpenAD remains unchanged. The small einops dependency is in pilot/python_deps, not installed into the OpenAD environment.

## Measured checks

- Full PartField vision encoder (106,890,240 parameters), random weights, 2,048 synthetic input points: CPU passed with finite output of shape [1, 2048, so 448].
- Same encoder on MPS: passed.
- Espresso heatmap decoder on MPS, consuming the encoder output and synthetic text features: passed with finite logits of shape [1, 2048].
- Latest combined MPS test elapsed about 3.50 seconds, including construction. This is not pretrained inference latency. No memory peak was measured.

Machine uses the existing PyTorch 2.14.1 interpreter. See espresso_cpu_check.json and espresso_mps_check.json for saved evidence; test_espresso_device.py for the test. CPU result covers the encoder only; latest MPS result covers encoder and synthetic-text decoder. No claim of CPU decoder verification.

## Interpretation and remaining work

The inspected inference path is not inherently blocked by CUDA-only native operators: the tested vision encoder and decoder ran locally after device/precision changes. Earlier statements that no local execution path was verified should now be updated to PARTIAL compatibility verified.

Not yet tested: real Recap-CLIP text tower/tokenizer, released Espresso/PartField weights, checkpoint compatibility, full model memory, 16,384-point input, real annotation input, accuracy, or CPU/MPS/CUDA numerical agreement. Random-weight outputs have no semantic meaning. No new pretrained affordance prediction has been generated.

Next milestone: isolate the full dependency stack, obtain/checkpoint-pin the released weights, and run one real object/query through the complete model. Then validate mapping and metrics before a pilot comparison. An architecture feasibility check alone must not be reported as Affogato experimental results. Affostruction's separate CUDA dependencies remain unresolved.

Run the compatibility check from the workspace root:

```sh
PYTHONPATH=pilot/python_deps /opt/homebrew/Caskroom/miniforge/base/envs/openad/bin/python3 pilot/test_espresso_device.py --device mps
```

## Update: complete pretrained inference passed

On 2026-10-05 the real Recap-CLIP text encoder, released PartField weights and unofficial Espresso Affogato-pretrained weights loaded successfully. One complete inference on OpenAD partial-view Vase sample 452 (2,048 points), with query 'Point to the part you would use to pour.', produced finite scores on MPS. Inference after model loading took 1.643 seconds; this is a single cold measurement, not a runtime benchmark. Current MPS allocation after inference was 1,794,523,904 bytes; it is NOT peak memory or total process memory.

See espresso_real/run.json for recorded Hub revisions, input identity and score range; espresso_real/heatmap.npy for newly computed scores; espresso_real/prediction.png for visualization. The real-input blocker listed above is now resolved for this one case. The 10-object comparison, numerical equivalence, and quantitative grounding evaluation remain pending. This remains an unofficial implementation, tested on an out-of-protocol input, not official Affogato reproduction. Affostruction has not run.

Dependencies are isolated in espresso_env with inherited read-only access to existing PyTorch packages. Added yacs for metadata deserialization in the PartField checkpoint. espresso_environment.txt records packages installed locally in the new environment. Large model downloads are confined to pilot/hf_cache.
