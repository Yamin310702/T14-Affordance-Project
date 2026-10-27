# Mac and MPS reproducibility audit

Audited 2026-10-11 against the saved experiment register, run metadata, result reports, and the active `pilot/espresso_env` interpreter. The user's interactive Terminal output is also considered evidence where it was pasted into the research conversation.

## What has actually run

- **OpenAD:** the original baseline and one-sample CPU/MPS checks are archived. The full dataset analysis completed for 9,140 views with 12.5364% mIoU. These are OpenAD results, not results for Affogato, Espresso, or Affostruction.
- **Espresso:** this is an *unofficial reimplementation* using Affogato-pretrained weights, adapted locally for MPS. A complete pretrained inference ran on one OpenAD Vase partial-view sample (2,048 points). Later prompt-sensitivity, held-out confirmation, mixed-presence, threshold-calibration, and E16 operating-point experiments are saved. They are exploratory cross-dataset evaluations, not official Affogato benchmark reproductions.
- **E16 on MPS:** the user's Terminal command used `--require-mps`, reported `device: mps`, processed the calibration and fresh test sets, and ended `Final status: complete`. Its paired CPU run has the same IDs and labels. Across 400 shared score maps, maximum absolute CPU–MPS difference was 6.29e-6; all maps were within `atol=rtol=1e-5`. The thresholds and reported conclusions matched. MPS does not materially change this E16 result.
- **Affostruction:** no inference or evaluation has run. The local checkout is source only; no Affostruction checkpoint/data preparation or experiment result is recorded. Its checked-in README specifies Linux x86_64 with CUDA 12.4, and the shown inference path calls `.cuda()`. A Mac MPS run is therefore not currently an available, validated setup.
- **Three-model comparison:** not run. The ten-object manifest is a plan, not evidence that all three methods produced results.

The detailed experiment-by-experiment record remains in [the archive index](README.md), [the 2026-10-11 experiment index](2026-10-11/README.md), and the dated result reports.

## What the Mac currently has for the validated Espresso/OpenAD path

- Apple Silicon / arm64 Mac; the project interpreter reports macOS 14.4, Python 3.10.21, and PyTorch 2.14.1.
- The working interpreter is `/Users/yaminmyataung/Documents/ChatGPT/3D Vision Research/pilot/espresso_env/bin/python`.
- That venv has `einops==0.8.2` and the project dependencies in `pilot/espresso_environment.txt`. It uses `include-system-site-packages = true` and inherits PyTorch from `/opt/homebrew/Caskroom/miniforge/base/envs/openad`. Thus it is a local overlay, not a portable standalone environment; changing/removing the `openad` Conda environment can break it.
- Local model cache is `pilot/hf_cache` (4.2 GB at audit time); the overlay is 168 MB. OpenAD data is under `/Users/yaminmyataung/Open-Vocabulary-Affordance-Detection-in-3D-Point-Clouds/data`.
- No additional package or computer is needed to repeat the already-tested E16 MPS experiment on this same setup, as long as the preflight below passes and there is enough free disk space for outputs/cache.

### Important device-check discrepancy

The E16 MPS run metadata records `device=mps` and `mps_available=true`, and the user's pasted Terminal output confirms MPS was used. In this audit's shell, the same venv currently reports `mps_built=True` but `mps_available=False`. This audit shell may not expose the same usable MPS runtime as the user's interactive Terminal. Therefore the successful user-terminal E16 log is evidence that the Mac ran it; the audit shell's status must not be mistaken for proof that the physical Mac lacks MPS. Always use the preflight in the same Terminal session that launches the experiment, and require the runner's `--require-mps` option.

**Follow-up profile:** the OpenAD sample-452 profiling script subsequently ran successfully in an MPS-enabled execution context on this Mac (`mps_available=true`). It measured a 0.5456 s synchronized forward after setup and a 5 ms sampled Metal-driver allocation high-water of 1,390,460,928 bytes. This resolves the practical device question for the tested sample, while the allocation remains a sampled estimate rather than a guaranteed hardware peak.

```sh
cd "/Users/yaminmyataung/Documents/ChatGPT/3D Vision Research"
./pilot/espresso_env/bin/python -c 'import sys, torch; print(sys.executable); print(torch.__version__); print("MPS built:", torch.backends.mps.is_built()); print("MPS available:", torch.backends.mps.is_available())'
```

For an MPS experiment, stop if `MPS available` is false; do not silently accept a CPU fallback. The E16 runner already has `--require-mps` for this purpose. Continue to use the project interpreter rather than bare `python3`, which can resolve to a different environment.

## Remaining requirements by goal

1. **Repeat or extend Espresso/OpenAD tests on this Mac:** keep the active `openad` Conda environment and `pilot/espresso_env`; run the preflight; use the project interpreter and `--require-mps`; preserve a fresh output directory and record `run.json`. No new GPU purchase is indicated by the completed experiments. The environment is tied to this Mac's existing Conda install and absolute paths.
2. **Run Affostruction's released inference as documented:** obtain access to a Linux x86_64 machine with CUDA 12.4 and a compatible NVIDIA GPU, then create its own environment and fetch its checkpoints/data. The README does not establish a minimum inference VRAM. Do not infer a memory requirement from its 8-GPU training recipe. The M3 Pro has no NVIDIA CUDA device, and this checkout has not been ported or validated for MPS.
3. **Make a fair Affogato comparison:** first make sure each method receives the same object/query cases and compatible input condition, define output/point mapping and metrics before running, and retain a genuinely held-out set. Existing OpenAD labels and the unofficial Espresso checkpoint do not by themselves constitute the official Affogato benchmark.

## Metadata correction

The MPS E16 `run.json` had a stale copied sentence saying that the run used CPU, contradicting its `device=mps`, `mps_available=true`, the CPU–MPS comparison, and the user's successful `--require-mps` terminal log. The sentence has been corrected in the working and archived JSON records; no predictions or metrics were changed.
