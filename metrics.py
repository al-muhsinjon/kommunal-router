"""Confusion matrix rows = actual, columns = predicted; zero division = 0."""
import numpy as np
from model import LABELS


def evaluate(actual, predicted):
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    indices = {label: i for i, label in enumerate(LABELS)}
    for truth, guess in zip(actual, predicted):
        matrix[indices[truth], indices[guess]] += 1
    tp = matrix.diagonal().astype(float)
    precision = tp / np.maximum(matrix.sum(axis=0), 1)
    recall = tp / np.maximum(matrix.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {"accuracy": float(tp.sum() / max(matrix.sum(), 1)),
            "macro_f1": float(f1.mean()), "labels": LABELS.tolist(),
            "per_class": {str(label): {"precision": float(p), "recall": float(r),
                                      "f1": float(f), "support": int(n)}
                          for label, p, r, f, n in zip(LABELS, precision, recall, f1, matrix.sum(axis=1))},
            "confusion_matrix": matrix.tolist()}
