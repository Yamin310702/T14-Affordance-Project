#!/usr/bin/env python3
"""Paired OpenAD comparison: training-name vs held-out query vocabulary.

Uses the released PN2 checkpoint and official AffordanceNet validation data.
Each point cloud is passed through the point backbone once; the resulting
features are scored against both vocabularies, so the pair shares geometry
features and PointNet++ sampling exactly.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/Users/yaminmyataung/Open-Vocabulary-Affordance-Detection-in-3D-Point-Clouds")
CHECKPOINT = REPO / "checkpoints/best_model_openad_pn2_partial.t7"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "research_archive/2026-10-11/seen_query_comparison"

TRAIN_QUERIES = [
    "grasp", "contain", "lift", "openable", "layable", "sittable",
    "support", "wrap_grasp", "pourable", "move", "displaY", "pushable",
    "pull", "listen", "wear", "press", "cut", "stab", "none",
]
HELDOUT_QUERIES = [
    "grab", "accommodate", "raise", "unlock", "rest", "take a seat", "bear",
    "wrap", "pour", "reposition", "demonstrate", "push", "drag", "hear",
    "clothe", "thumb", "slice", "jab", "none",
]
CONDITIONS = ("training_name_queries", "heldout_query_names")


def same_device(actual, expected):
    """Compare devices while treating unindexed MPS as MPS device 0."""
    import torch
    actual = torch.device(actual)
    expected = torch.device(expected)
    if actual.type != expected.type:
        return False
    if actual.type == "mps":
        return actual.index in (None, 0) and expected.index in (None, 0)
    return actual == expected


def point_features(model, xyz):
    """OpenAD PN2 geometry path through the learned point feature projection."""
    import torch
    xyz = xyz.contiguous()
    l0_xyz = xyz
    l0_points = xyz
    l1_xyz, l1_points = model.sa1(l0_xyz, l0_points)
    l2_xyz, l2_points = model.sa2(l1_xyz, l1_points)
    l3_xyz, l3_points = model.sa3(l2_xyz, l2_points)
    l2_points = model.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
    l1_points = model.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
    l0_points = model.fp1(l0_xyz, l1_xyz, torch.cat([l0_xyz, l0_points], 1), l1_points)
    l0_points = model.bn1(model.conv1(l0_points))
    return l0_points.permute(0, 2, 1).float()


def score(features, text_features, logit_scale):
    # The OpenAD forward path applies the same cosine similarity and log-softmax.
    import torch
    denom = torch.norm(features, dim=2, keepdim=True) @ torch.norm(text_features, dim=0, keepdim=True)
    logits = (logit_scale * (features @ text_features) / denom).permute(0, 2, 1)
    return torch.argmax(logits, dim=1)


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=("partial", "full"), required=True)
    p.add_argument("--limit", type=int, default=0, help="0 evaluates the complete validation split")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--device", choices=("cpu", "mps"), default="cpu",
                   help="inference device; MPS requests fail rather than falling back")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    # Resolve relative destinations from the caller's working directory before
    # changing into the external OpenAD source repository below.
    output_root = args.output_dir.expanduser().resolve()

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    sys.path.insert(0, str(REPO))
    os.chdir(REPO)

    import torch
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS was requested but torch.backends.mps.is_available() is False. Check the Python environment/PyTorch build; no CPU fallback was used.")
    device = torch.device(args.device)
    from torch.utils.data import DataLoader, Subset
    from dataset import AffordNetDataset
    from models.openad_pn2 import OpenAD_PN2, cls_encoder

    torch.set_num_threads(args.threads)
    np.random.seed(1)
    torch.manual_seed(1)
    encoder_device = torch.device(cls_encoder.device)
    if not same_device(encoder_device, device):
        raise RuntimeError(f"OpenAD text encoder is on {encoder_device}, but --device requested {device}.")
    partial = args.split == "partial"
    data_root = REPO / "data"
    out = output_root / f"{args.split}_validation"
    out.mkdir(parents=True, exist_ok=True)

    meta = {
        "status": "loading", "experiment": "paired_seen_query_vocabulary_comparison",
        "split": args.split, "data": "official AffordanceNet validation data",
        "checkpoint": str(CHECKPOINT), "checkpoint_sha256": None,
        "device": str(device), "mps_available": bool(torch.backends.mps.is_available()),
        "batch_size": args.batch_size, "threads": args.threads,
        "limit": args.limit, "training_name_queries": TRAIN_QUERIES,
        "heldout_query_names": HELDOUT_QUERIES,
        "pairing": "same input point cloud and one shared point-feature extraction per pair; both query vocabularies have aligned class slots",
        "metrics": "global point-wise IoU per class; macro over 19 classes (official-style, includes none); macro over 18 affordances (excludes none); point accuracy",
        "claim_limit": "This compares the config-derived training-name vocabulary with the config-derived held-out-query vocabulary. It does not by itself establish unseen-concept generalization because many names are semantic aliases.",
    }
    (out / "run.json").write_text(json.dumps(meta, indent=2) + "\n")

    import hashlib
    h = hashlib.sha256()
    with CHECKPOINT.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    meta["checkpoint_sha256"] = h.hexdigest()

    try:
        dataset = AffordNetDataset(str(data_root), "val", partial=partial)
        all_indices = list(range(len(dataset)))
        if args.limit > 0:
            all_indices = all_indices[:args.limit]
        loader = DataLoader(Subset(dataset, all_indices), batch_size=args.batch_size,
                            shuffle=False, num_workers=0, drop_last=False)
        meta.update(status="running", dataset_views=len(dataset), evaluated_views=len(all_indices),
                    unique_shape_ids=len({d["shape_id"] for d in dataset.all_data}))
        (out / "run.json").write_text(json.dumps(meta, indent=2) + "\n")

        model = OpenAD_PN2({}, len(TRAIN_QUERIES)).to(device).eval()
        state = torch.load(CHECKPOINT, map_location=device)
        model.load_state_dict(state, strict=True)
        cls_encoder.eval()
        with torch.inference_mode():
            text_train = cls_encoder(TRAIN_QUERIES)
            text_heldout = cls_encoder(HELDOUT_QUERIES)
        if not same_device(text_train.device, device) or not same_device(text_heldout.device, device):
            raise RuntimeError(f"Text features must be on {device}; got {text_train.device} and {text_heldout.device}.")

        confusion = {c: np.zeros((19, 19), dtype=np.int64) for c in CONDITIONS}
        object_rows = []
        start = time.monotonic()
        done = 0
        with torch.inference_mode():
            for batch_i, batch in enumerate(loader):
                data, _, labels, ids, categories = batch
                data = data.float().to(device).permute(0, 2, 1)
                labels_np = labels.squeeze(-1).cpu().numpy().astype(np.int64)
                feats = point_features(model, data)
                pred = {
                    CONDITIONS[0]: score(feats, text_train, model.logit_scale).cpu().numpy(),
                    CONDITIONS[1]: score(feats, text_heldout, model.logit_scale).cpu().numpy(),
                }
                for j in range(len(ids)):
                    gt = labels_np[j].reshape(-1)
                    idx = all_indices[done + j]
                    rowmeta = dataset.all_data[idx]
                    for cond in CONDITIONS:
                        pr = pred[cond][j].reshape(-1)
                        confusion[cond] += np.bincount(gt * 19 + pr, minlength=19 * 19).reshape(19, 19)
                        tp = np.bincount(gt[gt == pr], minlength=19)
                        actual = np.bincount(gt, minlength=19)
                        predicted = np.bincount(pr, minlength=19)
                        union = actual + predicted - tp
                        ious = np.divide(tp, union, out=np.zeros(19, dtype=float), where=union > 0)
                        object_rows.append({
                            "shape_id": str(ids[j]), "category": str(categories[j]),
                            "view_id": str(rowmeta.get("view_id", "full")),
                            "condition": cond,
                            "point_accuracy": float(np.mean(gt == pr)),
                            "miou_19_including_none": float(ious.mean()),
                            "miou_18_excluding_none": float(ious[:18].mean()),
                        })
                done += len(ids)
                if done % max(args.batch_size * 50, 1) == 0 or done == len(all_indices):
                    elapsed = time.monotonic() - start
                    print(f"{args.split}: {done}/{len(all_indices)} views ({elapsed:.1f}s)", flush=True)

        global_rows = []
        summary = {}
        for cond in CONDITIONS:
            cm = confusion[cond]
            tp = np.diag(cm).astype(np.float64)
            union = cm.sum(0) + cm.sum(1) - tp
            ious = np.divide(tp, union, out=np.zeros(19, dtype=np.float64), where=union > 0)
            for i in range(19):
                global_rows.append({"condition": cond, "class_index": i,
                                    "query_name": (TRAIN_QUERIES if cond == CONDITIONS[0] else HELDOUT_QUERIES)[i],
                                    "is_none": i == 18, "intersection": int(tp[i]), "union": int(union[i]),
                                    "iou": float(ious[i])})
            summary[cond] = {
                "miou_19_including_none": float(ious.mean()),
                "miou_18_excluding_none": float(ious[:18].mean()),
                "point_accuracy": float(tp.sum() / max(cm.sum(), 1)),
                "class_ious": ious.tolist(),
                "confusion_matrix": cm.tolist(),
            }
        deltas = {}
        for metric in ("miou_19_including_none", "miou_18_excluding_none", "point_accuracy"):
            deltas[metric] = (summary[CONDITIONS[1]][metric] - summary[CONDITIONS[0]][metric])
        meta.update(status="complete", elapsed_seconds=time.monotonic() - start,
                    summary=summary, heldout_minus_training_vocabulary=deltas,
                    output_files=["run.json", "per_class_iou.csv", "per_view_metrics.csv"])
        write_csv(out / "per_class_iou.csv", global_rows,
                  ["condition", "class_index", "query_name", "is_none", "intersection", "union", "iou"])
        write_csv(out / "per_view_metrics.csv", object_rows,
                  ["shape_id", "category", "view_id", "condition", "point_accuracy", "miou_19_including_none", "miou_18_excluding_none"])
    except Exception as exc:
        meta.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (out / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
        print("Final status:", meta["status"], flush=True)
    print(json.dumps({k: meta[k] for k in ("status", "split", "evaluated_views", "elapsed_seconds", "summary", "heldout_minus_training_vocabulary") if k in meta}, indent=2), flush=True)


if __name__ == "__main__":
    main()
