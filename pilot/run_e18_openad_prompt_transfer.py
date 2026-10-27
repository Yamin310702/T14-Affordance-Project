#!/usr/bin/env python3
"""E18: transfer five preselected Espresso prompt pairs to frozen OpenAD."""
from __future__ import annotations

import csv
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = Path("/Users/yaminmyataung/Open-Vocabulary-Affordance-Detection-in-3D-Point-Clouds")
OUT = ROOT / "research_archive/2026-10-11/E18_openad_prompt_transfer"
CHECKPOINT = REPO / "checkpoints/best_model_openad_pn2_partial.t7"
DATA_ROOT = REPO / "data"

VAL_QUERIES = [
    "grab", "accommodate", "raise", "unlock", "rest", "take a seat", "bear",
    "wrap", "pour", "reposition", "demonstrate", "push", "drag", "hear",
    "clothe", "thumb", "slice", "jab", "none",
]
PROMPTS = {
    "grab": {
        "class_index": 0,
        "original_sentence": "Point to the part you would grab.",
        "selected_paraphrase": "Point to the part you would grasp.",
    },
    "accommodate": {
        "class_index": 1,
        "original_sentence": "Point to the part of the object that can hold something.",
        "selected_paraphrase": "Point to the part where you would place or keep something.",
    },
    "unlock": {
        "class_index": 3,
        "original_sentence": "Point to the part you would use to unlock this object.",
        "selected_paraphrase": "Point to the part you would operate to gain access.",
    },
    "wrap": {
        "class_index": 7,
        "original_sentence": "Point to the part you can wrap your hand around.",
        "selected_paraphrase": "Point to the region designed to be held by wrapping your hand around it.",
    },
    "pour": {
        "class_index": 8,
        "original_sentence": "Point to the part you would use to pour.",
        "selected_paraphrase": "Point to the part used for pouring.",
    },
}
PRIOR_RUNS = [
    ROOT / "research_archive/2026-10-08/E09_multobject_query_sensitivity/run.json",
    ROOT / "research_archive/2026-10-08/E10_heldout_query_sensitivity/run.json",
    ROOT / "research_archive/2026-10-11/E11_grab_query_sensitivity/run.json",
    ROOT / "research_archive/2026-10-11/E12_multiaffordance_prompt_study/run.json",
    ROOT / "research_archive/2026-10-11/E13_heldout_prompt_confirmation/run.json",
]
CONDITIONS = ("canonical_word", "original_sentence", "selected_paraphrase")


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


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_ci(deltas: np.ndarray, rng: np.random.Generator, n_boot=10000):
    if len(deltas) == 0:
        return [None, None]
    samples = np.empty(n_boot, dtype=np.float64)
    # Chunk to keep memory bounded if the test set is expanded later.
    for start in range(0, n_boot, 500):
        n = min(500, n_boot - start)
        picks = rng.integers(0, len(deltas), size=(n, len(deltas)))
        samples[start:start+n] = deltas[picks].mean(axis=1)
    return [float(x) for x in np.quantile(samples, [0.025, 0.975])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-shapes", type=int, default=0, help="0 evaluates every eligible shape")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu",
                        help="inference device; MPS requests fail rather than falling back")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    # Resolve relative destinations from the caller's working directory before
    # changing into the external OpenAD source repository below.
    out_dir = args.output_dir.expanduser().resolve()
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(ROOT / "pilot"))
    os.chdir(REPO)

    import torch
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS was requested but torch.backends.mps.is_available() is False. Check the Python environment/PyTorch build; no CPU fallback was used.")
    device = torch.device(args.device)
    from dataset import AffordNetDataset
    from models.openad_pn2 import OpenAD_PN2, cls_encoder
    from run_seen_query_comparison import point_features
    from sklearn.metrics import roc_auc_score

    torch.set_num_threads(4)
    encoder_device = torch.device(cls_encoder.device)
    if not same_device(encoder_device, device):
        raise RuntimeError(f"OpenAD text encoder is on {encoder_device}, but --device requested {device}.")
    torch.manual_seed(1)
    np.random.seed(1)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_path = out_dir / "run.json"

    def selected_ids(obj):
        if isinstance(obj, list):
            return {x["shape_id"] for x in obj}
        if isinstance(obj, dict):
            return {x["shape_id"] for rows in obj.values() for x in rows}
        return set()

    excluded = set()
    for path in PRIOR_RUNS:
        meta = json.loads(path.read_text(encoding="utf-8"))
        excluded.update(selected_ids(meta.get("objects", {})))

    meta = {
        "experiment": "E18_openad_prompt_transfer",
        "status": "loading",
        "protocol": str(OUT / "PROTOCOL.md"),
        "model": "released OpenAD PN2 partial-view checkpoint; frozen",
        "checkpoint": str(CHECKPOINT),
        "device": str(device),
        "mps_available": bool(torch.backends.mps.is_available()),
        "seed": 1,
        "batching": "four partial views from one shape per shared geometry forward",
        "data": "official OpenAD validation partial views; exclude every shape ID selected in E09-E13",
        "prior_run_files_excluded": [str(p) for p in PRIOR_RUNS],
        "excluded_prior_prompt_study_shape_count": len(excluded),
        "canonical_heldout_vocabulary": VAL_QUERIES,
        "conditions": list(CONDITIONS),
        "prompts": PROMPTS,
        "intervention": "For one target class at a time, replace only its canonical held-out word with either its original sentence or frozen E13 paraphrase; all other class text stays fixed.",
        "primary_metric": "Global point-wise IoU per target class from the 19-way argmax across all retained views.",
        "secondary_metrics": ["positive-shape IoU", "positive-shape ROC AUC when defined", "negative-shape false-positive point fraction", "negative-shape any-alarm rate"],
        "uncertainty": "Paired shape-level bootstrap (10,000 resamples; seed 20261011), comparing selected paraphrase with original sentence.",
        "claim_limit": "Held-out objects within the official validation split; exploratory cross-model transfer from Espresso prompt selection. Not an official test set or proof of unseen affordance generalization.",
        "max_shapes_for_run": args.max_shapes,
    }
    h = hashlib.sha256()
    with CHECKPOINT.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    meta["checkpoint_sha256"] = h.hexdigest()
    run_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    try:
        dataset = AffordNetDataset(str(DATA_ROOT), "val", partial=True)
        groups = {}
        for idx, row in enumerate(dataset.all_data):
            sid = row["shape_id"]
            if sid in excluded:
                continue
            groups.setdefault(sid, []).append(idx)
        if args.max_shapes > 0:
            groups = dict(list(groups.items())[:args.max_shapes])
        shape_ids = list(groups)
        model = OpenAD_PN2({}, len(VAL_QUERIES)).to(device).eval()
        model.load_state_dict(torch.load(CHECKPOINT, map_location=device), strict=True)
        cls_encoder.eval()

        vocabularies = {("baseline", "all"): VAL_QUERIES}
        for affordance, spec in PROMPTS.items():
            class_index = spec["class_index"]
            for condition in CONDITIONS[1:]:
                words = list(VAL_QUERIES)
                words[class_index] = spec[condition]
                vocabularies[(affordance, condition)] = words
        text_features = {}
        with torch.inference_mode():
            text_features = {key: cls_encoder(words) for key, words in vocabularies.items()}
        wrong_devices = {str(t.device) for t in text_features.values() if not same_device(t.device, device)}
        if wrong_devices:
            raise RuntimeError(f"Text features must be on {device}; found {sorted(wrong_devices)}.")

        meta.update(status="running", dataset_views=len(dataset),
                    validation_shape_count=len({d["shape_id"] for d in dataset.all_data}),
                    selected_shape_count=len(shape_ids), evaluated_views=sum(map(len, groups.values())),
                    positive_and_negative_shape_counts={})
        (out_dir / "excluded_prior_shape_ids.txt").write_text("\n".join(sorted(excluded)) + "\n", encoding="utf-8")
        (out_dir / "selected_test_shape_ids.txt").write_text("\n".join(shape_ids) + "\n", encoding="utf-8")

        object_rows = []
        manifest = []
        global_counts = {a: {c: {k: 0 for k in ("tp", "fp", "fn", "points", "positive_points")} for c in CONDITIONS}
                         for a in PROMPTS}
        start = time.monotonic()
        rng = np.random.default_rng(20261011)

        with torch.inference_mode():
            for shape_n, sid in enumerate(shape_ids, 1):
                inds = groups[sid]
                samples = [dataset[i] for i in inds]
                xyz_np = np.stack([x[0] for x in samples]).astype(np.float32)
                gt_np = np.stack([x[2].reshape(-1) for x in samples]).astype(np.int64)
                xyz = torch.from_numpy(xyz_np).permute(0, 2, 1).to(device)
                feat = point_features(model, xyz)

                # Baseline vocabulary is scored once for all five target classes.
                feats_norm = torch.norm(feat, dim=2, keepdim=True)
                def logits_for(key):
                    text = text_features[key]
                    denom = feats_norm @ torch.norm(text, dim=0, keepdim=True)
                    logits = (model.logit_scale * (feat @ text) / denom)
                    return torch.softmax(logits, dim=-1).cpu().numpy()

                probs_by_key = {("baseline", "all"): logits_for(("baseline", "all"))}
                for affordance in PROMPTS:
                    for condition in CONDITIONS[1:]:
                        probs_by_key[(affordance, condition)] = logits_for((affordance, condition))

                category = str(samples[0][4])
                manifest_row = {"shape_id": sid, "category": category, "view_count": len(inds)}
                for affordance, spec in PROMPTS.items():
                    ci = spec["class_index"]
                    gt = gt_np.reshape(-1) == ci
                    npos = int(gt.sum())
                    manifest_row[f"gt_points_{affordance}"] = npos
                    manifest_row[f"has_{affordance}"] = bool(npos)
                    per_cond = {}
                    for condition in CONDITIONS:
                        key = ("baseline", "all") if condition == "canonical_word" else (affordance, condition)
                        probs = probs_by_key[key]
                        pred_cls = probs.argmax(axis=-1).reshape(-1)
                        pred = pred_cls == ci
                        score_vec = probs[..., ci].reshape(-1)
                        tp = int(np.count_nonzero(pred & gt))
                        fp = int(np.count_nonzero(pred & ~gt))
                        fn = int(np.count_nonzero(~pred & gt))
                        union = tp + fp + fn
                        iou = float(tp / union) if union else 0.0
                        auc = None
                        if npos > 0 and npos < gt.size:
                            auc = float(roc_auc_score(gt.astype(np.uint8), score_vec))
                        false_fraction = float(pred.mean()) if npos == 0 else None
                        alarm = bool(pred.any()) if npos == 0 else None
                        per_cond[condition] = {
                            "iou": iou, "auc": auc, "false_fraction": false_fraction,
                            "alarm": alarm, "tp": tp, "fp": fp, "fn": fn,
                            "points": int(gt.size), "positive_points": npos,
                        }
                        counts = global_counts[affordance][condition]
                        counts["tp"] += tp; counts["fp"] += fp; counts["fn"] += fn
                        counts["points"] += int(gt.size); counts["positive_points"] += npos
                        object_rows.append({
                            "shape_id": sid, "category": category, "affordance": affordance,
                            "class_index": ci, "condition": condition, "views": len(inds),
                            "ground_truth_points": npos, "total_points": int(gt.size),
                            "object_iou": iou, "object_auc": "" if auc is None else auc,
                            "negative_object_false_positive_fraction": "" if false_fraction is None else false_fraction,
                            "negative_object_any_alarm": "" if alarm is None else alarm,
                            "tp": tp, "fp": fp, "fn": fn,
                        })
                    # Keep paired per-shape metric records for bootstrap summaries.
                    manifest_row[f"iou_{affordance}_canonical"] = per_cond["canonical_word"]["iou"]
                    manifest_row[f"iou_{affordance}_original"] = per_cond["original_sentence"]["iou"]
                    manifest_row[f"iou_{affordance}_paraphrase"] = per_cond["selected_paraphrase"]["iou"]
                    manifest_row[f"fpr_{affordance}_canonical"] = per_cond["canonical_word"]["false_fraction"]
                    manifest_row[f"fpr_{affordance}_original"] = per_cond["original_sentence"]["false_fraction"]
                    manifest_row[f"fpr_{affordance}_paraphrase"] = per_cond["selected_paraphrase"]["false_fraction"]
                    manifest_row[f"alarm_{affordance}_original"] = per_cond["original_sentence"]["alarm"]
                    manifest_row[f"alarm_{affordance}_paraphrase"] = per_cond["selected_paraphrase"]["alarm"]
                manifest.append(manifest_row)

                if shape_n % 50 == 0 or shape_n == len(shape_ids):
                    elapsed = time.monotonic() - start
                    meta.update(completed_shapes=shape_n, elapsed_seconds=elapsed)
                    run_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
                    print(f"E18: {shape_n}/{len(shape_ids)} shapes ({elapsed:.1f}s)", flush=True)

        summaries = []
        paired = {}
        object_lookup = {(r["shape_id"], r["affordance"], r["condition"]): r for r in object_rows}
        for affordance, spec in PROMPTS.items():
            class_summaries = {}
            for condition in CONDITIONS:
                c = global_counts[affordance][condition]
                union = c["tp"] + c["fp"] + c["fn"]
                class_summaries[condition] = {
                    "global_iou": c["tp"] / union if union else 0.0,
                    "tp": c["tp"], "fp": c["fp"], "fn": c["fn"],
                    "points": c["points"], "positive_points": c["positive_points"],
                }
            for condition in CONDITIONS:
                rr = [r for r in object_rows if r["affordance"] == affordance and r["condition"] == condition]
                pos = [r for r in rr if int(r["ground_truth_points"]) > 0]
                neg = [r for r in rr if int(r["ground_truth_points"]) == 0]
                aucs = [float(r["object_auc"]) for r in pos if r["object_auc"] != ""]
                class_summaries[condition].update({
                    "positive_shapes": len(pos), "negative_shapes": len(neg),
                    "mean_positive_shape_iou": float(np.mean([float(r["object_iou"]) for r in pos])) if pos else None,
                    "mean_positive_shape_auc": float(np.mean(aucs)) if aucs else None,
                    "mean_negative_false_positive_fraction": float(np.mean([float(r["negative_object_false_positive_fraction"]) for r in neg])) if neg else None,
                    "negative_shape_any_alarm_rate": float(np.mean([str(r["negative_object_any_alarm"]).lower() == "true" for r in neg])) if neg else None,
                })
            # Paired cluster bootstrap over object IDs for the primary selected-vs-original contrasts.
            for metric_name, row_field, pool in [
                ("positive_shape_iou", "object_iou", "positive"),
                ("negative_shape_false_positive_fraction", "negative_object_false_positive_fraction", "negative"),
            ]:
                by = {r["shape_id"]: r for r in object_rows if r["affordance"] == affordance}
                if pool == "positive":
                    ids = [sid for sid in by if int(by[sid]["ground_truth_points"]) > 0]
                else:
                    ids = [sid for sid in by if int(by[sid]["ground_truth_points"]) == 0]
                def val(sid, cond):
                    row = object_lookup[(sid, affordance, cond)]
                    return float(row[row_field])
                deltas = np.asarray([val(sid, "selected_paraphrase") - val(sid, "original_sentence") for sid in ids], dtype=float)
                paired[f"{affordance}:{metric_name}"] = {
                    "n_shapes": len(ids), "mean_delta_paraphrase_minus_original": float(deltas.mean()) if len(deltas) else None,
                    "bootstrap_95pct_ci": bootstrap_ci(deltas, rng) if len(deltas) else [None, None],
                    "fraction_improved": float(np.mean(deltas > 0)) if len(deltas) else None,
                }
            summaries.append({"affordance": affordance, "class_index": spec["class_index"],
                              "canonical_heldout_query": VAL_QUERIES[spec["class_index"]],
                              "conditions": class_summaries})

        positive_negative_counts = {
            a: {"positive_shapes": sum(bool(x[f"has_{a}"]) for x in manifest),
                "negative_shapes": sum(not bool(x[f"has_{a}"]) for x in manifest)}
            for a in PROMPTS
        }
        meta.update(status="complete", elapsed_seconds=time.monotonic() - start,
                    positive_and_negative_shape_counts=positive_negative_counts,
                    per_affordance_summary=summaries, paired_bootstrap=paired,
                    output_files=["run.json", "per_affordance_summary.csv", "per_object_metrics.csv", "object_manifest.csv", "selected_test_shape_ids.txt", "excluded_prior_shape_ids.txt"])
        write_csv(out_dir / "per_object_metrics.csv", object_rows,
                  ["shape_id", "category", "affordance", "class_index", "condition", "views", "ground_truth_points", "total_points", "object_iou", "object_auc", "negative_object_false_positive_fraction", "negative_object_any_alarm", "tp", "fp", "fn"])
        write_csv(out_dir / "object_manifest.csv", manifest, list(manifest[0]) if manifest else ["shape_id"])
        summary_rows = []
        for item in summaries:
            for condition, vals in item["conditions"].items():
                summary_rows.append({"affordance": item["affordance"], "class_index": item["class_index"],
                                     "condition": condition, **vals})
        fields = sorted({k for r in summary_rows for k in r})
        write_csv(out_dir / "per_affordance_summary.csv", summary_rows, fields)
    except Exception as exc:
        meta.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        raise
    finally:
        run_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print("Final status:", meta["status"], flush=True)
    print(json.dumps({"status": meta["status"], "elapsed_seconds": meta.get("elapsed_seconds"),
                      "selected_shape_count": meta.get("selected_shape_count"),
                      "positive_and_negative_shape_counts": meta.get("positive_and_negative_shape_counts"),
                      "per_affordance_summary": meta.get("per_affordance_summary"),
                      "paired_bootstrap": meta.get("paired_bootstrap")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
