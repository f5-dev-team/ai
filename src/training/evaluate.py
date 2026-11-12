import numpy as np
from scipy.special import expit
from sklearn.metrics import (
    f1_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred

    probabilities = expit(logits)

    predictions = (
            probabilities >= 0.5
    ).astype(int)

    labels = labels.astype(int)

    metrics = {
        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            labels,
            predictions,
            average="macro",
            zero_division=0,
        ),
    }

    try:
        metrics["roc_auc"] = roc_auc_score(
            labels,
            probabilities,
            average="macro",
        )
    except ValueError:
        metrics["roc_auc"] = float("nan")

    return metrics