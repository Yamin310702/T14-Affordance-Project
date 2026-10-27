# Re-run E17 and E18 on Mac MPS

This guide runs the paired seen-query comparison (E17) and the fixed-prompt transfer experiment (E18) on Apple MPS. The earlier CPU results remain intact. MPS outputs go into separate folders. Both runners now accept `--device mps` and stop with an error if MPS is unavailable; they never silently switch to CPU.

## 1. Open Terminal and check the exact Python environment

Use the `openad` environment that you used for the earlier Mac experiments:

```zsh
cd "/Users/yaminmyataung/Documents/ChatGPT/3D Vision Research"
conda activate openad
export OPENAD_PYTHON="/opt/homebrew/Caskroom/miniforge/base/envs/openad/bin/python"
"$OPENAD_PYTHON" - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())
assert torch.backends.mps.is_available(), "MPS is unavailable in this Python environment. Stop here and check the environment."
PY
```

Continue only when `MPS available: True`. The experiment commands also enforce this check.

## 2. Keep the Mac awake and avoid overlapping runs

Plug in the Mac. In a second Terminal window, run `caffeinate -dimsu` while experiments are running; press Control-C in that window when done. Run one experiment command at a time so the jobs do not compete for unified memory.

## 3. Smoke-test both scripts on MPS

These short checks verify model loading, the MPS device, and the scoring path before a long run. Their temporary results go under `/private/tmp`.

```zsh
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
"$OPENAD_PYTHON" pilot/run_seen_query_comparison.py --split partial --limit 8 --batch-size 4 --threads 4 --device mps --output-dir /private/tmp/openad_e17_mps_smoke
"$OPENAD_PYTHON" pilot/run_e18_openad_prompt_transfer.py --max-shapes 2 --device mps --output-dir /private/tmp/openad_e18_mps_smoke
```

Each should end with `Final status: complete`. If either reports an MPS error, stop and save that terminal output; do not remove `--device mps` because that would run on CPU.

## 4. Run the complete E17 seen-query comparison

Run each split separately. Each command evaluates all eligible data for that split. Batch size 4 matches the archived E17 protocol.

```zsh
"$OPENAD_PYTHON" pilot/run_seen_query_comparison.py --split partial --batch-size 4 --threads 4 --device mps --output-dir "research_archive/2026-10-11/seen_query_comparison_mps"
```

After that finishes, run:

```zsh
"$OPENAD_PYTHON" pilot/run_seen_query_comparison.py --split full --batch-size 4 --threads 4 --device mps --output-dir "research_archive/2026-10-11/seen_query_comparison_mps"
```

The partial and full results are saved in `partial_validation/` and `full_validation/` within that output folder. Each contains `run.json`, `per_class_iou.csv`, and `per_view_metrics.csv`.

## 5. Run the complete E18 fixed-prompt transfer experiment

```zsh
"$OPENAD_PYTHON" pilot/run_e18_openad_prompt_transfer.py --device mps --output-dir "research_archive/2026-10-11/E18_openad_prompt_transfer/mps"
```

This evaluates all eligible shapes while excluding the same prior prompt-study shapes as the CPU run. Results are saved separately under `E18_openad_prompt_transfer/mps/`, including `run.json`, `per_affordance_summary.csv`, `per_object_metrics.csv`, and `object_manifest.csv`.

## 6. Confirm completion and device

For each finished run, open its `run.json` and check that `"status": "complete"` and `"device": "mps"`. E17 paths:

- `research_archive/2026-10-11/seen_query_comparison_mps/partial_validation/run.json`
- `research_archive/2026-10-11/seen_query_comparison_mps/full_validation/run.json`

E18 path:

- `research_archive/2026-10-11/E18_openad_prompt_transfer/mps/run.json`

The reported MPS-versus-CPU metric differences are device-reproduction checks. Small changes can come from numerical precision and point sampling; compare both splits and per-class results, and do not treat an MPS gain alone as a new research finding.
