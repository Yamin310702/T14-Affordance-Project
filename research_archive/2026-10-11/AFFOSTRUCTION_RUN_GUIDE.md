# Affostruction: practical run guide for this project

## Recommended experiment

For T14, start with the official **partial-input Affogato protocol**. It uses one RGB-D observation, reconstructs the object's 3D structure, predicts the requested affordance on that reconstruction, and evaluates against Affogato annotations. Do not train the model for the proposal. Use the released checkpoints and a subset of the official Affogato test manifest first.

This is a separate benchmark result from OpenAD: the input representation, dataset, reconstruction stage, and metrics differ. Do not compare its aIoU/aCD directly to OpenAD's point-wise mIoU unless you re-evaluate both on a shared protocol.

## Hardware decision

The official repository says it requires Linux x86_64 + CUDA 12.4, pins a Linux-only Python environment, and uses CUDA extensions. The M3 Pro and the teammate's Intel Iris Xe machine cannot run that released setup. Previous MPS success for OpenAD/Espresso does not make Affostruction's CUDA path MPS-compatible.

If you get temporary access to a Linux NVIDIA machine, use the official instructions below. A small GPU quota is enough for a pilot because this uses pretrained checkpoints; the official full benchmark contains 1,000 objects and can be run later if time/storage allow. The repository does not promise a minimum GPU memory size, so validate a single object before committing to the 20-object pilot.

If you cannot get such a machine before the proposal, stop at the Mac-side preparation section and report Affostruction as related work with no local reproduction. Do not attempt to relabel the teammate's point-cloud plot as an Affostruction prediction.

## Linux CUDA setup (only when a compatible machine is available)

On the CUDA machine, first clone the official repository and follow its locked environment setup:

```bash
git clone https://github.com/chrockey/Affostruction.git
cd Affostruction
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

Verify that the environment sees CUDA before downloading large checkpoints or data:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda); assert torch.cuda.is_available()"
```

Then run the repository's included single-sample demo as a smoke test:

```bash
uv run python examples/affostruction.py --data_dir examples/data/sample2
```

The first run downloads Affostruction and TRELLIS pretrained weights. Allow enough disk space and record the checkpoint IDs/revisions and file hashes if available. Do not train either stage for the proposal.

## Prepare a reproducible 20-object pilot

Follow the repository's `DATASET.md` and data-toolkit instructions to prepare the Affogato TRELLIS-format inputs. You need the official test split, processed `trellis_data/affogato` files (including `renders_cond` and affordance annotations), and the raw annotation files expected by the setup. Do not create a random split from training objects.

Make a deterministic 20-object subset from the official **test** manifest:

```bash
uv run python - <<'PY'
from pathlib import Path
import random

source = Path("manifests/affogato_test.txt")
target = Path("manifests/affogato_test_pilot20.txt")
rows = [line.strip() for line in source.read_text().splitlines() if line.strip()]
random.Random(20261011).shuffle(rows)
target.write_text("\n".join(rows[:20]) + "\n")
print(f"Wrote {target}: {min(20, len(rows))} of {len(rows)} official test objects")
PY
```

Run one-view reconstruction. Set `DATA_ROOT` to the folder containing the `affogato/` directory prepared by the official toolkit:

```bash
DATA_ROOT=/absolute/path/to/trellis_data
uv run python predict.py \
  --ckpt chrockey/Affostruction \
  --datasets affogato \
  --manifest manifests/affogato_test_pilot20.txt \
  --data_root "$DATA_ROOT" \
  --output_base results/affogato_pilot \
  --num_views 1 --seed 1
```

This writes reconstructions under `results/affogato_pilot/affogato/Affostruction/1/`. Check the log for errors and verify that all 20 objects have `outputs.npz` before running metrics.

Evaluate the partial affordance protocol using the repository defaults (one query per object, `query_idx=0`):

```bash
uv run python eval.py \
  --task partial \
  --ckpt chrockey/Affostruction \
  --data_dir "$DATA_ROOT/affogato" \
  --manifest manifests/affogato_test_pilot20.txt \
  --pred_dir results/affogato_pilot/affogato/Affostruction/1 \
  --output results/affogato_pilot/partial20_metrics.json \
  --per_sample --seed 0
```

This reports the official partial-protocol aIoU and aCD over available object/query pairs. If all five queries are scientifically relevant, make a separate secondary run with `--all_queries`; do not mix it with the one-query result or claim direct comparability unless the target published protocol also uses all five.

## What to save for a proposal-grade result

Keep the exact repository commit, environment/PyTorch/CUDA versions, GPU name and memory, checkpoint identity, manifest subset, sampler settings and seeds, logs, per-object outputs, metrics JSON, and failures. Report how many of the requested objects completed; never silently drop failed or missing objects. A 20-object subset is a feasibility pilot, not a benchmark reproduction. If it works and time permits, run the full official 1,000-object test protocol without tuning on it.

For the 8-minute proposal, use the pilot only if completed before the deadline and describe it as a pilot with its sample count. Otherwise state plainly: “Affostruction is a related method; its official CUDA inference could not be reproduced on our available hardware.” Cite the paper/repository's published values separately from your own OpenAD/Espresso experiments.

## What you can do now on the Mac

1. Keep the local source revision and the completed compatibility preflight in the archive.
2. Inspect Affogato `xyzc.npy`/`queries.json` pairs and verify channel-to-query ordering against the official toolkit, but label this as dataset inspection only.
3. Add a slide explaining Affostruction's two stages (partial RGB-D → reconstructed 3D geometry → text-conditioned affordance) and its published partial-input result with a citation.
4. Put a clear “not reproduced locally” note next to published metrics. Do not list them in the table of your experimental results.

## Official references

- [Affostruction code, setup, model usage and benchmark protocols](https://github.com/chrockey/Affostruction)
- [Released pretrained checkpoints](https://huggingface.co/chrockey/Affostruction)
- [Affostruction paper](https://arxiv.org/abs/2601.09211)
