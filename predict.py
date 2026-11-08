"""Usage: python predict.py input.csv output.csv; model path is CWD-independent."""
import argparse
from pathlib import Path
import pandas as pd
from dataio import read_rows
from model import load, route
from text import is_urgent


def main():
    parser = argparse.ArgumentParser(description="Kommunal murojaatlarni yo'naltirish")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        rows = read_rows(args.input)
        vectorizer, model = load(Path(__file__).resolve().with_name("model.npz"))
        output = []
        for start in range(0, len(rows), 256):
            batch = rows[start:start + 256]
            labels = route(model, vectorizer, [row["text"] for row in batch])
            output.extend((row["id"], label, is_urgent(row["text"])) for row, label in zip(batch, labels))
        pd.DataFrame(output, columns=["id", "label", "is_urgent"]).to_csv(args.output, index=False, encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
