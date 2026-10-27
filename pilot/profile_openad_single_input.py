#!/usr/bin/env python3
"""Profile one OpenAD validation sample on MPS without CPU fallback.

Run from the Mac's existing OpenAD environment. The sampled memory value is a
high-water observation of total Metal-driver allocation during the forward;
PyTorch MPS does not expose a CUDA-style peak-allocation counter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time


REPO = Path("/Users/yaminmyataung/Open-Vocabulary-Affordance-Detection-in-3D-Point-Clouds")
DEFAULT_CONFIG = "config/openad_pn2/partial_view_open_vocab_cfg.py"
DEFAULT_CHECKPOINT = "checkpoints/best_model_openad_pn2_partial.t7"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "openad_sample452_mps_profile.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--index", type=int, default=452)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-ms", type=float, default=5.0,
                        help="Interval for sampling Metal-driver allocation during inference.")
    args = parser.parse_args()

    repo = args.repo.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not repo.is_dir():
        raise SystemExit(f"OpenAD repository not found: {repo}")
    sys.path.insert(0, str(repo))
    os.chdir(repo)

    import numpy as np
    import torch
    from gorilla.config import Config
    from dataset.AffordanceNet import AffordNetDataset
    from utils import build_model, set_random_seed

    if not torch.backends.mps.is_built() or not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available in this Python process; refusing CPU fallback.")
    device = torch.device("mps")
    torch.manual_seed(1)
    np.random.seed(1)

    config_path = (repo / args.config).resolve()
    checkpoint_path = (repo / args.checkpoint).resolve()
    cfg = Config.fromfile(str(config_path))
    dataset = AffordNetDataset(cfg.data.data_root, "val", partial=cfg.training_cfg.partial)
    if not (0 <= args.index < len(dataset)):
        raise SystemExit(f"Index {args.index} outside validation dataset of {len(dataset)} samples")
    xyz, _, labels, shape_id, category = dataset[args.index]
    affordances = list(cfg.training_cfg.val_affordance)
    model = build_model(cfg).to(device)
    model.eval()
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    torch.mps.synchronize()

    points = torch.from_numpy(np.asarray(xyz, dtype=np.float32)).unsqueeze(0)
    points = points.permute(0, 2, 1).to(device)
    label = np.asarray(labels).reshape(-1).astype(np.int64)
    base_alloc = int(torch.mps.driver_allocated_memory())
    peak_sampled = [base_alloc]
    stop = threading.Event()

    def sample_memory() -> None:
        delay = max(args.sample_ms, 1.0) / 1000.0
        while not stop.is_set():
            try:
                peak_sampled[0] = max(peak_sampled[0], int(torch.mps.driver_allocated_memory()))
            except Exception:
                pass
            stop.wait(delay)

    monitor = threading.Thread(target=sample_memory, daemon=True)
    monitor.start()
    torch.mps.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        scores = model(points, affordances)
    torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    stop.set()
    monitor.join()

    prediction = scores.argmax(dim=1).squeeze(0).cpu().numpy()
    per_class_iou = {}
    for class_id, name in enumerate(affordances):
        gt = label == class_id
        pred = prediction == class_id
        union = int(np.logical_or(gt, pred).sum())
        if union:
            per_class_iou[name] = float(np.logical_and(gt, pred).sum() / (union + 1e-6))
        else:
            per_class_iou[name] = 0.0
    miou = float(np.mean(list(per_class_iou.values())))
    after_alloc = int(torch.mps.driver_allocated_memory())
    current_tensor_alloc = int(torch.mps.current_allocated_memory())

    record = {
        "status": "complete",
        "model": "OpenAD_PN2",
        "device": "mps",
        "device_note": "Apple M3 Pro integrated GPU; unified memory, not a separate VRAM pool.",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "sample_index": args.index,
        "shape_id": str(shape_id),
        "category": str(category),
        "points": int(points.shape[-1]),
        "ground_truth_point_counts": {str(i): int((label == i).sum()) for i in np.unique(label)},
        "inference_seconds_excluding_model_and_data_setup": elapsed,
        "mps_driver_allocated_before_bytes": base_alloc,
        "mps_driver_allocated_after_bytes": after_alloc,
        "mps_driver_allocated_sampled_high_water_bytes": int(peak_sampled[0]),
        "mps_driver_sampling_interval_ms": args.sample_ms,
        "mps_current_tensor_allocated_after_bytes": current_tensor_alloc,
        "memory_note": "Sampled driver allocation is an observed high-water estimate, not a guaranteed hardware peak. MPS has no separate VRAM pool.",
        "sample_miou": miou,
        "per_class_iou": per_class_iou,
        "seed": 1,
        "point_count_note": "The dataset sample and normalization follow the repository loader; no extra resampling was added by this profiler.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
