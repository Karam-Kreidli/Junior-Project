"""Training-time metric helpers."""

from sklearn.metrics import average_precision_score
from sklearn.preprocessing import label_binarize


def compute_map(y_true, y_scores, num_classes):
    """Macro mAP over `num_classes`; 0.0 if it can't be computed."""
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    if y_true_bin.shape[1] <= 1:
        return 0.0
    try:
        return average_precision_score(y_true_bin, y_scores, average="macro")
    except Exception:
        return 0.0
