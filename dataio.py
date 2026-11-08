"""Pandas CSV input with literal IDs/text, BOM support, and schema validation."""
import pandas as pd


def read_rows(path, labeled=False):
    required = {"id", "text", "label"} if labeled else {"id", "text"}
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig", engine="python")
    if not required.issubset(frame.columns):
        raise ValueError(f"{path}: required columns: {', '.join(sorted(required))}")
    if not isinstance(frame.index, pd.RangeIndex) or frame[sorted(required)].isna().any().any():
        raise ValueError(f"{path}: invalid CSV row length")
    return frame.to_dict(orient="records")
