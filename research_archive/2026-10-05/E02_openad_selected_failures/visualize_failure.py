import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from gorilla.config import Config

from utils import build_model, build_dataset


def compute_iou(pred, gt, class_id):
    pred_mask = pred == class_id
    gt_mask = gt == class_id

    intersection = np.sum(pred_mask & gt_mask)
    union = np.sum(pred_mask | gt_mask)

    if union == 0:
        return None

    return intersection / union


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="./config/openad_pn2/partial_view_open_vocab_cfg.py"
    )

    parser.add_argument(
        "--checkpoint",
        default="./checkpoints/best_model_openad_pn2_partial.t7"
    )

    parser.add_argument(
        "--affordance",
        default="clothe"
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # 1. Load configuration
    # ---------------------------------------------------------
    cfg = Config.fromfile(args.config)

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print("Using device:", device)

    # ---------------------------------------------------------
    # 2. Build OpenAD
    # ---------------------------------------------------------
    model = build_model(cfg).to(device)

    print("Loading checkpoint:", args.checkpoint)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device
    )

    model.load_state_dict(checkpoint)
    model.eval()

    # ---------------------------------------------------------
    # 3. Load exactly the same validation dataset
    # ---------------------------------------------------------
    dataset_dict = build_dataset(cfg)
    val_set = dataset_dict["val_set"]

    affordances = list(cfg.training_cfg.val_affordance)

    print("\nValidation affordances:")
    for i, name in enumerate(affordances):
        print(i, name)

    if args.affordance not in affordances:
        raise ValueError(
            f"{args.affordance} is not in validation affordances."
        )

    target_id = affordances.index(args.affordance)

    print("\nTarget affordance:", args.affordance)
    print("Class ID:", target_id)
    print("Number of validation samples:", len(val_set))

    # ---------------------------------------------------------
    # 4. Search for low-IoU examples
    # ---------------------------------------------------------
    candidates = []

    with torch.no_grad():

        for index in range(len(val_set)):

            points, _, labels, model_id, model_cat = val_set[index]

            labels = np.asarray(labels).squeeze().astype(np.int64)

            # Only examine samples where GT actually contains
            # the target affordance.
            if not np.any(labels == target_id):
                continue

            x = torch.from_numpy(points).float()
            x = x.unsqueeze(0)
            x = x.permute(0, 2, 1)
            x = x.to(device)

            output = model(x, affordances)

            output = output.permute(0, 2, 1)
            output = output.detach().cpu().numpy()[0]

            pred = np.argmax(output, axis=1)

            iou = compute_iou(
                pred,
                labels,
                target_id
            )

            if iou is not None:

                candidates.append(
                    {
                        "index": index,
                        "iou": iou,
                        "points": points,
                        "labels": labels,
                        "pred": pred,
                        "model_id": model_id,
                        "model_cat": model_cat,
                    }
                )

    if len(candidates) == 0:
        print(
            "\nNo validation sample contains ground truth for:",
            args.affordance
        )
        return

    # Lowest IoU first
    candidates.sort(key=lambda x: x["iou"])

    print("\nLowest-IoU examples")
    print("-------------------")

    for item in candidates[:10]:
        print(
            "index =", item["index"],
            "| IoU =", round(item["iou"], 4),
            "| shape =", item["model_id"],
            "| category =", item["model_cat"]
        )

    # ---------------------------------------------------------
    # 5. Visualize the worst example
    # ---------------------------------------------------------
    worst = candidates[0]

    points = worst["points"]
    labels = worst["labels"]
    pred = worst["pred"]

    gt_mask = labels == target_id
    pred_mask = pred == target_id

    fig = plt.figure(figsize=(15, 5))

    # Raw partial point cloud
    ax1 = fig.add_subplot(131, projection="3d")

    ax1.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=4
    )

    ax1.set_title("Partial Point Cloud")

    # Ground truth
    ax2 = fig.add_subplot(132, projection="3d")

    ax2.scatter(
        points[~gt_mask, 0],
        points[~gt_mask, 1],
        points[~gt_mask, 2],
        s=3,
        alpha=0.15
    )

    ax2.scatter(
        points[gt_mask, 0],
        points[gt_mask, 1],
        points[gt_mask, 2],
        s=8
    )

    ax2.set_title(
        f"Ground Truth: {args.affordance}"
    )

    # OpenAD prediction
    ax3 = fig.add_subplot(133, projection="3d")

    ax3.scatter(
        points[~pred_mask, 0],
        points[~pred_mask, 1],
        points[~pred_mask, 2],
        s=3,
        alpha=0.15
    )

    ax3.scatter(
        points[pred_mask, 0],
        points[pred_mask, 1],
        points[pred_mask, 2],
        s=8
    )

    ax3.set_title(
        f"OpenAD Prediction\nIoU = {worst['iou']:.3f}"
    )

    # Use similar viewing angles.
    for ax in [ax1, ax2, ax3]:
        ax.view_init(elev=20, azim=45)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    fig.suptitle(
        f"Partial-view failure | "
        f"{args.affordance} | "
        f"category: {worst['model_cat']} | "
        f"shape: {worst['model_id']}"
    )

    plt.tight_layout()

    os.makedirs("failure_visualizations", exist_ok=True)

    output_path = os.path.join(
        "failure_visualizations",
        f"failure_{args.affordance}.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print("\nWorst example")
    print("-------------")
    print("Dataset index:", worst["index"])
    print("Shape ID:", worst["model_id"])
    print("Category:", worst["model_cat"])
    print("Affordance:", args.affordance)
    print("IoU:", round(worst["iou"], 4))

    print("\nSaved visualization:")
    print(output_path)


if __name__ == "__main__":
    main()

