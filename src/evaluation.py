import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


def evaluate(y_true, y_pred, classes):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    pc, rc, fc, sc = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "precision": p, "recall": r, "f1": f,
            "confusion": confusion_matrix(y_true, y_pred, labels=classes),
            "per_class": pd.DataFrame({"Category": classes, "Precision": pc, "Recall": rc, "F1": fc, "Support": sc}),
            "y_pred": np.asarray(y_pred)}


def select_best(results):
    """Highest Macro F1; ties broken by accuracy, then by training order."""
    return max(results, key=lambda m: (results[m]["f1"], results[m]["accuracy"]))


def comparison_table(results, best):
    return pd.DataFrame([{"Model Name": m, "Accuracy": r["accuracy"], "Macro Precision": r["precision"],
                          "Macro Recall": r["recall"], "Macro F1": r["f1"],
                          "Status": "Selected" if m == best else "Evaluated"} for m, r in results.items()])
