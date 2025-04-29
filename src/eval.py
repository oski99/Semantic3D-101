import os
import numpy as np
import logging
from collections import defaultdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("eval_metrics.txt", mode='w'),  # Save output to a file
        logging.StreamHandler()  # Also print to console
    ]
)

# Paths
gt_dir = 'Semantic3D/processed'
pred_dir = 'test/Semantic3D'

# List prediction files
pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.labels')]

# Storage for overall metrics
all_per_class_iou = defaultdict(list)
all_per_class_acc = defaultdict(list)
all_total_tp = 0
all_total_fp = 0
all_total_fn = 0
all_total_correct = 0
all_total_samples = 0

# Process each prediction file
for pred_file in pred_files:
    pred_path = os.path.join(pred_dir, pred_file)
    gt_path = os.path.join(gt_dir, pred_file)

    if not os.path.exists(gt_path):
        logging.warning(f"Ground truth file not found for {pred_file}, skipping...")
        continue

    # Load labels
    gt_labels = np.loadtxt(gt_path, dtype=np.int32)
    pred_labels = np.loadtxt(pred_path, dtype=np.int32)

    assert gt_labels.shape == pred_labels.shape, f"Shape mismatch for {pred_file}"

    # Mask to skip invalid labels (gt == 0)
    valid_mask = gt_labels != 0
    gt_labels = gt_labels[valid_mask]
    pred_labels = pred_labels[valid_mask]

    classes = np.unique(np.concatenate((gt_labels, pred_labels)))

    file_total_tp = 0
    file_total_fp = 0
    file_total_fn = 0

    for cls in classes:
        tp = np.sum((gt_labels == cls) & (pred_labels == cls))
        fp = np.sum((gt_labels != cls) & (pred_labels == cls))
        fn = np.sum((gt_labels == cls) & (pred_labels != cls))

        intersection = tp
        union = tp + fp + fn
        iou = intersection / union if union > 0 else 0.0
        acc = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        all_per_class_iou[cls].append(iou)
        all_per_class_acc[cls].append(acc)

        file_total_tp += tp
        file_total_fp += fp
        file_total_fn += fn

    all_total_tp += file_total_tp
    all_total_fp += file_total_fp
    all_total_fn += file_total_fn
    all_total_correct += np.sum(gt_labels == pred_labels)
    all_total_samples += len(gt_labels)
# Calculate final metrics
logging.info("Average Per-class IoU:")
per_class_ious = []
for cls in sorted(all_per_class_iou.keys()):
    avg_iou = np.mean(all_per_class_iou[cls])
    per_class_ious.append(avg_iou)  # Store for mIoU calculation
    logging.info(f"Class {cls}: {avg_iou:.4f}")

logging.info("Average Per-class Accuracy:")
per_class_accs = []
for cls in sorted(all_per_class_acc.keys()):
    avg_acc = np.mean(all_per_class_acc[cls])
    per_class_accs.append(avg_acc)  # Store for mean accuracy calculation
    logging.info(f"Class {cls}: {avg_acc:.4f}")

mean_acc = np.mean(per_class_accs) if per_class_accs else 0.0  # Mean of per-class accuracies
mean_iou = np.mean(per_class_ious) if per_class_ious else 0.0  # Mean of per-class IoUs (mIoU)

logging.info(f"Mean Accuracy: {mean_acc:.4f}")
logging.info(f"Mean IoU (mIoU): {mean_iou:.4f}")