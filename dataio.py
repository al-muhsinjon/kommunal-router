"""UTF-8 CSV input, including BOM support and clear schema errors."""
import csv


def read_rows(path, labeled=False):
    required = {"id", "text", "label"} if labeled else {"id", "text"}
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: required columns: {', '.join(sorted(required))}")
        rows = list(reader)
    if any(any(row.get(key) is None for key in required) for row in rows):
        raise ValueError(f"{path}: incomplete CSV row")
    return rows
