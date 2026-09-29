import re
from pathlib import Path
import pandas as pd

_DEV = re.compile(r"[\u0900-\u097F]")
MIN_PER_CLASS = 5


class DatasetError(Exception):
    pass


def detect_columns(df):
    """Text column = string column with the highest Devanagari share; label column = fewest unique values among the rest."""
    str_cols = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype).startswith("str")]
    if len(str_cols) < 2:
        raise DatasetError("Could not find a text column and a label column (need at least two string columns).")
    share = {c: df[c].dropna().astype(str).head(500).map(lambda s: bool(_DEV.search(s))).mean() for c in str_cols}
    text_col = max(share, key=share.get)
    if share[text_col] < 0.5:
        raise DatasetError("No column appears to contain Marathi (Devanagari) text.")
    label_col = min((c for c in str_cols if c != text_col), key=lambda c: df[c].nunique())
    return text_col, label_col


def load_dataset(path):
    p = Path(path)
    if not p.exists():
        raise DatasetError(f"Dataset not found at {p}. Place the CSV in the data/ folder.")
    raw = None
    for enc in ("utf-8", "utf-8-sig"):
        try:
            raw = pd.read_csv(p, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            raise DatasetError(f"Could not parse CSV: {e}")
    if raw is None:
        raise DatasetError("File is not valid UTF-8; re-save the CSV as UTF-8 to preserve Devanagari.")
    text_col, label_col = detect_columns(raw)
    notes, df = [], raw.copy()
    n_missing = int(df[[text_col, label_col]].isna().any(axis=1).sum())
    df = df.dropna(subset=[text_col, label_col])
    df[label_col] = df[label_col].astype(str).str.strip()
    df[text_col] = df[text_col].astype(str)
    df = df[df[text_col].str.strip() != ""]
    if n_missing:
        notes.append(f"{n_missing} rows with a missing text or label were excluded (in memory only).")
    n_dup = int(df.duplicated(subset=[text_col, label_col]).sum())
    if n_dup:
        df = df.drop_duplicates(subset=[text_col, label_col])
        notes.append(f"{n_dup} exact duplicate rows were removed to avoid train/test leakage (in memory only).")
    counts = df[label_col].value_counts()
    small = counts[counts < MIN_PER_CLASS].index.tolist()
    if small:
        df = df[~df[label_col].isin(small)]
        notes.append(f"Categories with fewer than {MIN_PER_CLASS} samples were excluded: {small}.")
    if df[label_col].nunique() < 2:
        raise DatasetError("Fewer than two usable categories remain.")
    df = df.reset_index(drop=True)
    stats = {"file": p.name, "raw_rows": len(raw), "rows": len(df), "n_classes": df[label_col].nunique(),
             "missing_text": int(raw[text_col].isna().sum()), "missing_label": int(raw[label_col].isna().sum()),
             "duplicates_text": int(raw[text_col].duplicated().sum()), "duplicates_rows": n_dup,
             "counts": df[label_col].value_counts(),
             "words": df[text_col].str.split().str.len().agg(["mean", "min", "max"]),
             "chars": df[text_col].str.len().agg(["mean", "min", "max"])}
    return {"df": df, "raw_columns": list(raw.columns), "text_col": text_col, "label_col": label_col,
            "stats": stats, "notes": notes}
