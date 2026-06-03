"""
Preprocessing pipeline for CIC-IDS2017 (per-flow classification).

WHY PER-FLOW (not sequences)?
-----------------------------
The CIC-IDS2017 "MachineLearningCSV" files do not include source-IP or
timestamp columns, so flows cannot be ordered in time or grouped by host. That
makes sliding-window "sequences" meaningless on this data - a window would just
be arbitrary adjacent rows from concatenated files. So we classify each flow on
its own. This is the standard approach for these CSVs and what almost all
published results on the dataset do.

ORDER OF OPERATIONS (defend this in an interview):
  1. Merge every CSV in data/raw/ into one frame.
  2. Clean: strip column-name whitespace, replace Inf with NaN, drop NaN rows,
     drop exact-duplicate flows.
  3. Select features: drop identifier/leakage columns, keep numeric features.
  4. Encode labels: collapse the ~15 raw labels into coarse families.
  5. SPLIT FIRST, THEN scale: fit the MinMaxScaler ON TRAINING FLOWS ONLY so no
     information from validation/test leaks into the model.
  6. Save the scaler, the label map, and the .npy arrays.

    python -m src.preprocess
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def _first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column in `candidates` that exists (handles naming
    differences between dataset exports)."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_raw(raw_dir: Path) -> pd.DataFrame:
    """STEP 1 - ingest: merge every CSV in data/raw/ into one DataFrame."""
    csvs = sorted(raw_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(
            f"No CSVs in {raw_dir}. Put CIC-IDS2017 files there, or run "
            f"`python -m src.synthetic_data` to generate a test file."
        )
    print(f"[load] merging {len(csvs)} CSV file(s)...")
    frames = [pd.read_csv(p, low_memory=False) for p in csvs]
    df = pd.concat(frames, ignore_index=True)
    # CIC-IDS2017 ships headers with leading spaces (" Flow Duration"); strip
    # them so every later column lookup works.
    df.columns = [c.strip() for c in df.columns]
    print(f"[load] {len(df):,} rows, {df.shape[1]} columns")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """STEP 2 - clean: remove values that would break or bias training."""
    before = len(df)
    # Rate features (e.g. 'Flow Bytes/s') divide by a duration that can be zero,
    # producing +/-Inf which explodes gradients. Convert Inf -> NaN, then drop.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    df = df.drop_duplicates()
    print(f"[clean] dropped {before - len(df):,} bad/duplicate rows -> {len(df):,} remain")
    return df


def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """STEP 3 - feature selection + leakage prevention.

    Drop identifier columns (IPs, ports, timestamps, flow IDs). Leaving the
    source IP in would let the model memorize attacker addresses instead of
    learning behavior, inflating accuracy in a way that collapses on new
    traffic. Then keep only numeric feature columns.
    """
    label_col = _first_present(df, config.LABEL_CANDIDATES)
    if label_col is None:
        raise KeyError("No 'Label' column found.")
    y = df[label_col].astype(str).str.strip()

    drop = {c for c in df.columns if c.strip().lower()
            in {d.strip().lower() for d in config.IDENTIFIER_COLUMNS}}
    drop.add(label_col)
    feature_df = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
    feature_df = feature_df.select_dtypes(include=[np.number])
    print(f"[features] kept {feature_df.shape[1]} numeric features")
    return feature_df, y


def encode_labels(y: pd.Series) -> tuple[np.ndarray, dict[str, int]]:
    """STEP 4 - collapse the ~15 raw labels into coarse families and integer-encode."""
    grouped = y.map(lambda lbl: config.LABEL_GROUPING.get(lbl, "Other"))
    classes = sorted(grouped.unique())            # sorted -> deterministic encoding
    label_map = {name: i for i, name in enumerate(classes)}
    encoded = grouped.map(label_map).to_numpy()
    counts = grouped.value_counts().to_dict()
    print(f"[labels] {len(classes)} classes: {counts}")
    return encoded, label_map


def split_and_scale(X: np.ndarray, y: np.ndarray):
    """STEPS 5 - split BEFORE scaling; fit the scaler on TRAIN ONLY.

    Stratified split so rare attack classes appear in train, val AND test.
    Fitting MinMax on the full dataset would leak test statistics into training
    and inflate every metric, so we fit on training flows only.
    """
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED,
        stratify=y if len(np.unique(y)) > 1 else None,
    )
    val_rel = config.VAL_SIZE / (1.0 - config.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_rel, random_state=config.RANDOM_SEED,
        stratify=y_tmp if len(np.unique(y_tmp)) > 1 else None,
    )

    # Data is 2D (samples, features) for per-flow classification.
    scaler = MinMaxScaler()
    scaler.fit(X_train)                            # <-- fit on train only
    X_train = scaler.transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    print(f"[split] train={len(X_train):,} val={len(X_val):,} test={len(X_test):,}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler


def main():
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw(config.RAW_DIR)
    df = clean(df)
    features, y_raw = select_features(df)
    labels, label_map = encode_labels(y_raw)

    X = features.to_numpy(dtype=np.float32)        # (n_samples, n_features)
    (Xtr, ytr), (Xva, yva), (Xte, yte), scaler = split_and_scale(X, labels)

    np.save(config.PROCESSED_DIR / "X_train.npy", Xtr)
    np.save(config.PROCESSED_DIR / "y_train.npy", ytr)
    np.save(config.PROCESSED_DIR / "X_val.npy", Xva)
    np.save(config.PROCESSED_DIR / "y_val.npy", yva)
    np.save(config.PROCESSED_DIR / "X_test.npy", Xte)
    np.save(config.PROCESSED_DIR / "y_test.npy", yte)
    joblib.dump(scaler, config.SCALER_PATH)        # reused identically at inference
    with open(config.LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"\n[done] arrays -> {config.PROCESSED_DIR}")
    print(f"[done] scaler -> {config.SCALER_PATH}")
    print(f"[done] label map -> {config.LABEL_MAP_PATH}")
    print(f"[done] feature count = {Xtr.shape[-1]} (update config.NUM_FEATURES if this differs)")


if __name__ == "__main__":
    main()
