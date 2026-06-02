"""
Preprocessing pipeline for CIC-IDS2017.

WHY THIS FILE EXISTS
--------------------
Raw network-flow CSVs are not something a neural network can consume directly:
they contain text columns, infinities, duplicates, identifier fields that would
let the model "cheat", and they aren't grouped into the time-ordered sequences
this project classifies. This module turns raw CSVs into clean, scaled,
sequence-shaped NumPy arrays and saves the transformers needed to repeat the
exact same processing at inference time.

THE ORDER OF OPERATIONS IS DELIBERATE (and is the thing to defend in an interview):
  1. Merge every CSV in data/raw/ into one frame.
  2. Clean: strip column-name whitespace, replace Inf with NaN, drop NaN rows,
     drop exact-duplicate flows.
  3. Select features: drop identifier/leakage columns, keep numeric features.
  4. Encode labels: collapse the ~15 raw labels into coarse families.
  5. Build sequences: order flows by time, group by source IP, slide a window
     of length SEQ_LEN -> tensors of shape (SEQ_LEN, NUM_FEATURES).
  6. SPLIT FIRST, THEN scale: fit the MinMaxScaler ON TRAINING FLOWS ONLY so no
     information from validation/test leaks into the model (the single most
     common silent bug in ML projects).
  7. Save the scaler, the label map, and the .npy arrays so training and the
     demo all consume identical, reproducible inputs.

Run with real data:   put CSVs in data/raw/, then `python -m src.preprocess`
Run a smoke test:      `python -m src.synthetic_data && python -m src.preprocess`
"""
import json
import sys
from pathlib import Path

import joblib                                   # persists the fitted scaler to disk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler  # scales every feature to [0, 1]

# Make `import config` work whether this is run as a module or a script.
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def _first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column name in `candidates` that actually exists.

    Different CIC-IDS2017 exports name the same field differently (e.g.
    'Source IP' vs 'Src IP'). Rather than hard-code one name and crash on the
    other, we probe a list of likely names and use whichever is present.
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_raw(raw_dir: Path) -> pd.DataFrame:
    """STEP 1 - ingest: merge every CSV in data/raw/ into a single DataFrame."""
    csvs = sorted(raw_dir.glob("*.csv"))
    if not csvs:
        # Fail loudly with an actionable message instead of a cryptic error later.
        raise FileNotFoundError(
            f"No CSVs in {raw_dir}. Put CIC-IDS2017 files there, or run "
            f"`python -m src.synthetic_data` to generate a test file."
        )
    print(f"[load] merging {len(csvs)} CSV file(s)...")
    # low_memory=False avoids pandas guessing column dtypes chunk-by-chunk,
    # which otherwise produces mixed-type-column warnings on large files.
    frames = [pd.read_csv(p, low_memory=False) for p in csvs]
    df = pd.concat(frames, ignore_index=True)
    # CIC-IDS2017 is infamous for shipping headers with a leading space
    # (" Flow Duration"). Stripping here means every later column lookup
    # ("Label", "Source IP", ...) works without surprise KeyErrors.
    df.columns = [c.strip() for c in df.columns]
    print(f"[load] {len(df):,} rows, {df.shape[1]} columns")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """STEP 2 - clean: remove values that would break or bias training."""
    before = len(df)
    # Rate features like 'Flow Bytes/s' divide by a duration that can be zero,
    # producing +/-Inf. Inf flowing into a network explodes the gradients and
    # silently wrecks training, so convert Inf -> NaN first...
    df = df.replace([np.inf, -np.inf], np.nan)
    # ...then drop any row containing NaN (from the step above or already present).
    df = df.dropna()
    # Exact-duplicate flows would let identical samples land in both train and
    # test, inflating scores. Drop them.
    df = df.drop_duplicates()
    # Printing how much we removed is a cheap sanity check: if cleaning deletes
    # most of the data, something upstream is wrong and we want to notice.
    print(f"[clean] dropped {before - len(df):,} bad/duplicate rows -> {len(df):,} remain")
    return df


def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, str | None, str | None]:
    """STEP 3 - feature selection + leakage prevention.

    Returns (feature_matrix, raw_labels, src_ip_col_name, timestamp_col_name).
    The src/timestamp names are handed back so build_sequences can order and
    group flows; they are NOT part of the feature matrix.
    """
    label_col = _first_present(df, config.LABEL_CANDIDATES)
    if label_col is None:
        raise KeyError("No 'Label' column found.")
    src_col = _first_present(df, config.SRC_IP_CANDIDATES)
    ts_col = _first_present(df, config.TIMESTAMP_CANDIDATES)

    # Pull the target out and normalize its text (some labels have stray spaces).
    y = df[label_col].astype(str).str.strip()

    # ---- THE KEY DECISION ----------------------------------------------------
    # Drop identifier columns (IPs, ports, timestamps, flow IDs). If we left the
    # source IP in, the model could simply memorize "attacks came from
    # 10.0.0.5" and post a fake-high accuracy that collapses on a new network.
    # Removing identifiers forces it to learn behavioral features (packet sizes,
    # timing, flag counts) that actually generalize. This is "leakage
    # prevention" #1.
    drop = {c for c in df.columns if c.strip().lower()
            in {d.strip().lower() for d in config.IDENTIFIER_COLUMNS}}
    drop.add(label_col)                              # the target is not a feature
    feature_df = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")
    # Anything still object-typed is leftover metadata, not a numeric feature.
    feature_df = feature_df.select_dtypes(include=[np.number])
    print(f"[features] kept {feature_df.shape[1]} numeric features")
    return feature_df, y, src_col, ts_col


def encode_labels(y: pd.Series) -> tuple[np.ndarray, dict[str, int]]:
    """STEP 4 - target preparation: collapse raw labels into coarse families.

    CIC-IDS2017 has ~15 raw labels, several with only a handful of samples
    (too few to learn). config.LABEL_GROUPING maps, e.g., 'DoS Hulk',
    'DoS GoldenEye', 'DoS slowloris' all to 'DoS'. Anything unmapped -> 'Other'.
    We also build and return the name->int map so predictions can be decoded
    back to human-readable class names later.
    """
    grouped = y.map(lambda lbl: config.LABEL_GROUPING.get(lbl, "Other"))
    # Sort the class names so the integer encoding is deterministic across runs
    # (important for reproducibility and for matching the saved label_map).
    classes = sorted(grouped.unique())
    label_map = {name: i for i, name in enumerate(classes)}
    encoded = grouped.map(label_map).to_numpy()
    counts = grouped.value_counts().to_dict()
    # The printed counts reveal the class imbalance we handle later with weights.
    print(f"[labels] {len(classes)} classes: {counts}")
    return encoded, label_map


def build_sequences(features: pd.DataFrame, labels: np.ndarray,
                    src: pd.Series | None, ts: pd.Series | None,
                    seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """STEP 5 - turn individual flows into time-ordered sequences.

    WHY SEQUENCES? A single flow in isolation often looks benign; an *attack*
    usually shows up as a pattern across consecutive flows from the same host
    (a port scan is many probes in a row, not one connection). So we slide a
    window of `seq_len` flows to build tensors of shape (n, seq_len, n_features).

    Grouping by source IP matters: mixing flows from different hosts into one
    window would be physically meaningless. A window's label is its LAST flow's
    label - we classify the most recent flow given the context that preceded it.
    """
    feat = features.to_numpy(dtype=np.float32)
    n_features = feat.shape[1]

    # Decide the ordering of flows before windowing.
    if src is not None and ts is not None:
        # Sort by timestamp, then group row indices by source IP. We carry the
        # original row index 'idx' so we can pull the matching feature rows.
        order_df = pd.DataFrame({"src": src.to_numpy(),
                                 "ts": pd.to_datetime(ts, errors="coerce").to_numpy()})
        order_df["idx"] = np.arange(len(order_df))
        groups = [g["idx"].to_numpy() for _, g in
                  order_df.sort_values("ts").groupby("src", sort=False)]
        print(f"[sequence] windowing within {len(groups)} source IPs")
    else:
        # Fallback for data without IP/timestamp: one contiguous group.
        groups = [np.arange(len(feat))]
        print("[sequence] no IP/timestamp -> contiguous windows")

    X_seq, y_seq = [], []
    for idx in groups:
        if len(idx) < seq_len:
            continue                                  # too short to form a window
        # Slide the window one flow at a time across this source IP's flows.
        for start in range(len(idx) - seq_len + 1):
            window = idx[start:start + seq_len]
            X_seq.append(feat[window])                # (seq_len, n_features)
            y_seq.append(labels[window[-1]])          # label = last flow's label

    if not X_seq:
        raise ValueError(
            f"No sequences produced. Need >= {seq_len} flows per source IP."
        )
    X = np.stack(X_seq).astype(np.float32)
    y = np.asarray(y_seq)
    print(f"[sequence] {X.shape[0]:,} sequences of shape {X.shape[1:]} ")
    return X, y


def split_and_scale(X: np.ndarray, y: np.ndarray):
    """STEPS 6+7 - split BEFORE scaling, fit the scaler on TRAIN ONLY.

    Two leakage-prevention decisions live here:
      (a) Stratified split so rare attack classes appear in train, val AND test.
          Without stratification, a random split on an 80%-benign dataset could
          leave a rare class out of the test set entirely.
      (b) Fit the scaler on training flows only, then apply it to val/test. If
          we fit on the full dataset, statistics from the test set leak into
          training and inflate every reported number. Letting val/test values
          fall slightly outside [0, 1] is correct and expected.
    """
    # First carve off the test set.
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED,
        stratify=y if len(np.unique(y)) > 1 else None,
    )
    # Then split the remainder into train/val. We rescale VAL_SIZE so it stays a
    # fraction of the *original* dataset, not of the leftover after test removal.
    val_rel = config.VAL_SIZE / (1.0 - config.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_rel, random_state=config.RANDOM_SEED,
        stratify=y_tmp if len(np.unique(y_tmp)) > 1 else None,
    )

    # MinMaxScaler expects 2D (samples, features); our data is 3D
    # (samples, seq_len, features). Reshape to (-1, features) to fit per-feature
    # across all timesteps, fit on TRAIN ONLY, then transform each split.
    n, seq_len, n_features = X_train.shape
    scaler = MinMaxScaler()
    scaler.fit(X_train.reshape(-1, n_features))       # <-- fit on train only

    def apply(arr):
        s = arr.shape
        return scaler.transform(arr.reshape(-1, n_features)).reshape(s).astype(np.float32)

    X_train, X_val, X_test = apply(X_train), apply(X_val), apply(X_test)
    print(f"[split] train={len(X_train):,} val={len(X_val):,} test={len(X_test):,}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler


def main():
    """Run the whole pipeline and persist every artifact (STEP 8)."""
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Stages run strictly in order; each consumes the previous one's output.
    df = load_raw(config.RAW_DIR)
    df = clean(df)
    features, y_raw, src_col, ts_col = select_features(df)
    labels, label_map = encode_labels(y_raw)

    # Hand the (optional) src/timestamp columns to the sequencer.
    src = df[src_col] if src_col else None
    ts = df[ts_col] if ts_col else None
    X, y = build_sequences(features, labels, src, ts, config.SEQ_LEN)

    (Xtr, ytr), (Xva, yva), (Xte, yte), scaler = split_and_scale(X, y)

    # Persist arrays so training never has to re-run preprocessing.
    np.save(config.PROCESSED_DIR / "X_train.npy", Xtr)
    np.save(config.PROCESSED_DIR / "y_train.npy", ytr)
    np.save(config.PROCESSED_DIR / "X_val.npy", Xva)
    np.save(config.PROCESSED_DIR / "y_val.npy", yva)
    np.save(config.PROCESSED_DIR / "X_test.npy", Xte)
    np.save(config.PROCESSED_DIR / "y_test.npy", yte)
    # The scaler is an ARTIFACT, not throwaway state: inference must scale new
    # data with this exact fitted scaler or the model sees a different range.
    joblib.dump(scaler, config.SCALER_PATH)
    # Save the label map so predicted integers can be turned back into names.
    with open(config.LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"\n[done] arrays -> {config.PROCESSED_DIR}")
    print(f"[done] scaler -> {config.SCALER_PATH}")
    print(f"[done] label map -> {config.LABEL_MAP_PATH}")
    print(f"[done] feature count = {Xtr.shape[-1]} (update config.NUM_FEATURES if this differs)")


if __name__ == "__main__":
    main()
