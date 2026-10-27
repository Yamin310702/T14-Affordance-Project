# Affostruction execution preflight — 2026-10-11

## Outcome

**No Affostruction inference or metric was produced.** This preflight checked whether the available local setup can run the released evaluation code. It cannot, so it is recorded as an environment compatibility check, not as an Affostruction experiment or result.

## Checks performed

| Check | Observation |
|---|---|
| Local source | `pilot/Affostruction`, Git revision `bd37f11` (“Release training, evaluation and data preparation code”). |
| Official environment declared by the repository | Linux x86_64 + CUDA 12.4; Python >=3.10,<3.11; pinned PyTorch 2.5.0 and CUDA-specific dependencies including `spconv-cu124`, CUDA-build `xformers`, Kaolin 0.18.0, `nvdiffrast`, and related packages. |
| Available runtime queried | macOS 14.4 arm64; the interpreter reached through `pilot/espresso_env/bin/python` reports PyTorch 2.14.1, `torch.cuda.is_available() == False`, and `torch.backends.mps.is_available() == False` (MPS is built into this PyTorch build). This disagrees with the earlier archived E12/E16 MPS runs, which did succeed; treat this as an interpreter/environment discrepancy, not proof that the Mac itself cannot use MPS. |
| Required packages checked | `spconv`, `xformers`, `kaolin`, and `nvdiffrast` are not installed in the active environment. |
| Official CLI smoke check | `python pilot/Affostruction/predict.py --help` exits before argument parsing with `ModuleNotFoundError: No module named 'rootutils'`. This identifies an incomplete environment; installing that one package would not resolve the CUDA/platform incompatibility. |
| Weights and data | No Affostruction checkpoint (`.safetensors`, `.pt`, `.pth`, `.ckpt`) is present in the repository. The official manifest lists 1,000 Affogato test objects, but no processed TRELLIS Affogato dataset root was found in the project/Downloads/Documents candidate locations checked. |

## Why a valid Mac run is not available here

The released prediction path calls `.cuda()` directly. Its locked environment is restricted to Linux x86_64, and it depends on CUDA 12.4 extensions. The available M3 Pro does not provide CUDA. Prior MPS success in the OpenAD/Espresso environment does not replace the missing CUDA extension stack or make this official inference path MPS-compatible. This preflight did not modify the upstream model or its evaluation protocol.

## Proposal treatment

Do not report an Affostruction score, visualization, or “failed model prediction” from this check. State that the official code and test manifest were inspected, but the released model could not be executed on the available Mac; use the official paper/repository results only as published related-work context. A valid local reproduction requires an x86_64 Linux environment with a supported NVIDIA CUDA 12.4 GPU, the released checkpoints, and processed Affogato/TRELLIS inputs. The teammate's Intel Iris Xe laptop does not meet that CUDA requirement either.

This result has been added to [the complete experiment evidence map](ALL_EXPERIMENTS_PROPOSAL_MAP.md) as an environment preflight. It does not change the existing OpenAD or Espresso/Affogato-weight quantitative results.
