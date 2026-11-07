"""Train-only stratified CV, final training, and a held-out validation report."""
import csv
import json
from pathlib import Path
import numpy as np
from dataio import read_rows
from metrics import evaluate
from model import LABELS, Softmax, Tfidf, route, save
from plot_loss import save_loss

ROOT = Path(__file__).resolve().parent


def fit(texts, labels, config):
    vectorizer = Tfidf().fit(texts)
    y = np.array([np.flatnonzero(LABELS == label)[0] for label in labels])
    return vectorizer, Softmax().fit(vectorizer.transform(texts), y, **config)


def errors(rows, predictions):
    return [dict(row, predicted=str(pred)) for row, pred in zip(rows, predictions) if row["label"] != pred]


def main():
    rows = read_rows(ROOT / "data/train.csv", labeled=True)
    texts, labels = np.array([r["text"] for r in rows]), np.array([r["label"] for r in rows])
    rng, folds = np.random.default_rng(42), np.empty(len(rows), dtype=int)
    for label in LABELS:
        indices = rng.permutation(np.flatnonzero(labels == label))
        folds[indices] = np.arange(len(indices)) % 3
    trials = []
    for lr in (0.5, 1.0):
        for l2 in (0.001, 0.01, 0.05):
            config, guesses = dict(lr=lr, l2=l2, epochs=600), np.empty(len(rows), dtype="U11")
            for fold in range(3):
                train, test = folds != fold, folds == fold
                vectorizer, model = fit(texts[train], labels[train], config)
                guesses[test] = route(model, vectorizer, texts[test])
            trials.append(dict(config=config, macro_f1=evaluate(labels, guesses)["macro_f1"],
                               errors=errors(rows, guesses)))
            print(config, "OOF macro-F1:", round(trials[-1]["macro_f1"], 4), flush=True)
    best = max(trials, key=lambda trial: trial["macro_f1"])
    vectorizer, model = fit(texts, labels, best["config"])
    save(ROOT / "model.npz", vectorizer, model)
    validation = read_rows(ROOT / "data/validation.csv", labeled=True)
    val_texts = [row["text"] for row in validation]
    raw, final = model.predict(vectorizer.transform(val_texts)), route(model, vectorizer, val_texts)
    actual = [row["label"] for row in validation]
    report = dict(config=best["config"], cv_seed=42, cv_folds=3, trials=trials,
                  validation=evaluate(actual, final), validation_raw=evaluate(actual, raw),
                  validation_errors=errors(validation, final), validation_raw_errors=errors(validation, raw),
                  initial_loss=model.losses[0], final_loss=model.losses[-1])
    (ROOT / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(ROOT / "loss.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "loss"])
        writer.writerows(enumerate(model.losses))
    save_loss(model.losses, ROOT / "loss.png")
    print(json.dumps(report["validation"], indent=2))


if __name__ == "__main__":
    main()
