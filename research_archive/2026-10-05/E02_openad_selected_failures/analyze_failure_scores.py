import numpy as np
import torch
from gorilla.config import Config
from utils import build_model, build_dataset


CONFIG = "./config/openad_pn2/partial_view_open_vocab_cfg.py"
CHECKPOINT = "./checkpoints/best_model_openad_pn2_partial.t7"

SAMPLE_INDEX = 452
TARGET = "pour"


cfg = Config.fromfile(CONFIG)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Using device:", device)


# --------------------------------------------------
# Load model
# --------------------------------------------------

model = build_model(cfg).to(device)

model.load_state_dict(
    torch.load(
        CHECKPOINT,
        map_location=device
    )
)

model.eval()


# --------------------------------------------------
# Load validation sample
# --------------------------------------------------

dataset = build_dataset(cfg)["val_set"]

points, _, labels, shape_id, category = dataset[SAMPLE_INDEX]

labels = np.asarray(labels).squeeze().astype(np.int64)

affordances = list(cfg.training_cfg.val_affordance)

target_id = affordances.index(TARGET)

print("\nSample information")
print("------------------")
print("Index:", SAMPLE_INDEX)
print("Shape ID:", shape_id)
print("Category:", category)
print("Target:", TARGET)
print("Target class ID:", target_id)


# --------------------------------------------------
# Run OpenAD
# --------------------------------------------------

x = torch.from_numpy(points).float()
x = x.unsqueeze(0)
x = x.permute(0, 2, 1)
x = x.to(device)

with torch.no_grad():

    output = model(
        x,
        affordances
    )

# [1, classes, points]
log_probs = output[0].permute(1, 0).cpu().numpy()

# Convert log-softmax back to probabilities
probs = np.exp(log_probs)

pred = np.argmax(probs, axis=1)


# --------------------------------------------------
# Examine GT target points only
# --------------------------------------------------

gt_mask = labels == target_id

print("\nGround-truth target points:", np.sum(gt_mask))

target_probs = probs[gt_mask]

target_pred = pred[gt_mask]


# --------------------------------------------------
# Average probability for every class
# on GT target points
# --------------------------------------------------

mean_probs = target_probs.mean(axis=0)

ranking = np.argsort(mean_probs)[::-1]


print("\nAverage prediction on GT target region")
print("--------------------------------------")

for rank, class_id in enumerate(ranking[:10], start=1):

    print(
        f"{rank:2d}. "
        f"{affordances[class_id]:15s} "
        f"{mean_probs[class_id]:.6f}"
    )


# --------------------------------------------------
# How often each class wins on GT target points
# --------------------------------------------------

unique, counts = np.unique(
    target_pred,
    return_counts=True
)

order = np.argsort(counts)[::-1]


print("\nArgmax classes on GT target region")
print("----------------------------------")

for idx in order:

    class_id = unique[idx]

    percentage = (
        counts[idx] /
        len(target_pred)
    ) * 100

    print(
        f"{affordances[class_id]:15s} "
        f"{counts[idx]:5d} points "
        f"({percentage:.2f}%)"
    )


# --------------------------------------------------
# Target statistics
# --------------------------------------------------

print("\nTarget statistics")
print("-----------------")

print(
    "Mean target probability:",
    float(target_probs[:, target_id].mean())
)

print(
    "Maximum target probability:",
    float(target_probs[:, target_id].max())
)

print(
    "Points predicted correctly:",
    int(np.sum(target_pred == target_id))
)

print(
    "Correct percentage:",
    float(
        np.mean(target_pred == target_id) * 100
    )
)
