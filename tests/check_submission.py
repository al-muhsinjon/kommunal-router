"""External CLI checks; answers never enter predict.py. Run from a clean clone."""
import argparse
from collections import Counter
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["billing", "electricity", "gas", "waste", "water"]


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def score(rows, guesses):
    counts = Counter((row["label"], guess["label"]) for row, guess in zip(rows, guesses))
    matrix = [[counts[a, b] for b in LABELS] for a in LABELS]
    per_class = {}
    for index, label in enumerate(LABELS):
        tp = matrix[index][index]
        precision = tp / max(sum(row[index] for row in matrix), 1)
        recall = tp / max(sum(matrix[index]), 1)
        per_class[label] = dict(precision=precision, recall=recall,
                               f1=2 * precision * recall / max(precision + recall, 1e-12))
    result = dict(rows=len(rows), correct=sum(row["label"] == guess["label"] for row, guess in zip(rows, guesses)),
                  labels=LABELS, per_class=per_class, confusion_matrix=matrix,
                  errors=[dict(row, predicted=guess["label"], predicted_urgent=guess["is_urgent"])
                          for row, guess in zip(rows, guesses) if row["label"] != guess["label"]])
    result.update(accuracy=result["correct"] / len(rows),
                  macro_f1=sum(row["f1"] for row in per_class.values()) / len(LABELS))
    if "is_urgent" in rows[0]:
        result["urgency_correct"] = sum(row["is_urgent"] == guess["is_urgent"] for row, guess in zip(rows, guesses))
        result["urgency_errors"] = [dict(row, predicted_urgent=guess["is_urgent"])
                                    for row, guess in zip(rows, guesses) if row["is_urgent"] != guess["is_urgent"]]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "examiner_results.local.json")
    args = parser.parse_args()
    model_path = ROOT / "model.npz"
    before = hashlib.sha256(model_path.read_bytes()).hexdigest()
    validation = read_csv(ROOT / "data/validation.csv")
    custom = read_csv(ROOT / "tests/examiner_cases.csv")
    with tempfile.TemporaryDirectory(dir=ROOT) as folder:
        work = Path(folder)
        hook = work / "network_block"
        hook.mkdir()
        (hook / "sitecustomize.py").write_text(
            "import socket\ndef blocked(*a, **k):\n raise RuntimeError('Network disabled for examiner check')\n"
            "socket.socket.connect = blocked\nsocket.socket.connect_ex = blocked\n"
            "socket.create_connection = blocked\nsocket.getaddrinfo = blocked\n", encoding="utf-8")
        env = dict(os.environ, PYTHONUTF8="1", PYTHONPATH=str(hook))
        # Prove the subprocess actually loads the network block before testing inference.
        probe = subprocess.run([sys.executable, "-c", "import socket; socket.create_connection(('example.com',443))"],
                               env=env, cwd=work, capture_output=True, text=True, timeout=10)
        assert probe.returncode != 0 and "Network disabled" in probe.stderr

        def predict(rows, name):
            source, target = work / f"{name}.csv", work / f"{name}_output.csv"
            with source.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "text"])
                writer.writeheader()
                writer.writerows({key: row[key] for key in ("id", "text")} for row in rows)
            start = time.perf_counter()
            process = subprocess.run([sys.executable, str(ROOT / "predict.py"), str(source), str(target)],
                                     cwd=work, env=env, capture_output=True, text=True, timeout=30)
            assert process.returncode == 0, process.stderr
            with target.open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                assert reader.fieldnames == ["id", "label", "is_urgent"]
                guesses = list(reader)
            assert len(guesses) == len(rows)
            assert [r["id"] for r in guesses] == [r["id"] for r in rows]
            assert all(r["label"] in LABELS and r["is_urgent"] in ("0", "1") for r in guesses)
            return guesses, round(time.perf_counter() - start, 3)

        val_guesses, val_seconds = predict(validation, "validation")
        custom_guesses, custom_seconds = predict(custom, "custom")
        large = [dict(custom[i % len(custom)], id=f"batch-{i:04d}") for i in range(513)]
        large_guesses, large_seconds = predict(large, "batch")
        assert all((r["label"], r["is_urgent"]) == (custom_guesses[i % len(custom)]["label"],
                   custom_guesses[i % len(custom)]["is_urgent"]) for i, r in enumerate(large_guesses))
    assert before == hashlib.sha256(model_path.read_bytes()).hexdigest(), "Inference changed the model"
    report = dict(python=platform.python_version(), numpy=importlib.metadata.version("numpy"),
                  pandas=importlib.metadata.version("pandas"), model_sha256=before,
                  conditions=dict(input_columns=["id", "text"], python_network_blocked=True,
                                  external_working_directory=True, model_unchanged=True, batch_rows=513),
                  validation=score(validation, val_guesses), custom=score(custom, custom_guesses),
                  cli_seconds=dict(validation=val_seconds, custom=custom_seconds, batch=large_seconds))
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for name in ("validation", "custom"):
        result = report[name]
        print(f"{name}: {result['correct']}/{result['rows']}; macro-F1={result['macro_f1']:.4f}")
        for error in result["errors"]:
            print(f"  {error['id']}: {error['label']} -> {error['predicted']}: {error['text']}")
    print(f"Custom urgency: {report['custom']['urgency_correct']}/{len(custom)}")
    print("PASS: schema, IDs/order, 513-row batching, network block, unchanged model")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
