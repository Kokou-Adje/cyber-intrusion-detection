"""
Generate synthetic raw CSVs shaped like CIC-IDS2017.

This is NOT a model and it does NOT fake results. It writes raw flow records
with the same structure as the real dataset (Source IP, Timestamp, ~78 numeric
features, a Label column) so you can exercise the *real* preprocessing,
training, and dashboard code before downloading the multi-gigabyte dataset.

Each attack family is given a slightly different feature distribution so the
models have a real (if easy) signal to learn. Swap data/raw/ for the genuine
CIC-IDS2017 CSVs and nothing downstream changes.

Usage:
    python -m src.synthetic_data            # writes one CSV to data/raw/
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


# A representative slice of the real CIC-IDS2017 feature names. We don't need
# all 78 to be authentic for a smoke test; the count is what matters for shape.
_BASE_FEATURE_NAMES = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std",
    "Fwd IAT Mean", "Bwd IAT Mean", "Fwd PSH Flags", "Bwd PSH Flags",
    "Fwd Header Length", "Bwd Header Length", "Min Packet Length",
    "Max Packet Length", "Packet Length Mean", "Packet Length Std",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
    "ACK Flag Count", "URG Flag Count", "Down/Up Ratio", "Average Packet Size",
    "Subflow Fwd Packets", "Subflow Bwd Packets", "Init_Win_bytes_forward",
    "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward",
    "Active Mean", "Idle Mean",
]


def _feature_names(n_features: int) -> list[str]:
    names = list(_BASE_FEATURE_NAMES)
    i = 0
    while len(names) < n_features:
        names.append(f"ExtraFeature_{i}")
        i += 1
    return names[:n_features]


def generate_raw_csv(
    n_rows: int = 12000,
    n_features: int = config.NUM_FEATURES,
    n_src_ips: int = 60,
    seed: int = config.RANDOM_SEED,
    out_path: Path | None = None,
) -> Path:
    rng = np.random.default_rng(seed)
    feature_names = _feature_names(n_features)

    # Families and their relative frequency (benign-heavy, like the real set).
    families = ["Benign", "DDoS", "DoS", "Probe", "WebAttack"]
    weights = np.array([0.70, 0.10, 0.08, 0.07, 0.05])

    # Each family draws from a different mean so a model can separate them.
    family_means = {fam: rng.uniform(-1.0, 1.0, size=n_features) * (i + 1)
                    for i, fam in enumerate(families)}

    # A raw label name per family so preprocessing's LABEL_GROUPING is tested.
    family_to_raw = {
        "Benign": "BENIGN", "DDoS": "DDoS", "DoS": "DoS Hulk",
        "Probe": "PortScan", "WebAttack": "Web Attack \u2013 Brute Force",
    }

    chosen = rng.choice(families, size=n_rows, p=weights)
    X = np.vstack([
        rng.normal(loc=family_means[fam], scale=1.0) for fam in chosen
    ])
    # Real CIC-IDS2017 contains occasional Inf/NaN; inject a few so the
    # cleaning step actually has something to clean.
    n_bad = max(1, n_rows // 500)
    bad_rows = rng.integers(0, n_rows, size=n_bad)
    bad_cols = rng.integers(0, n_features, size=n_bad)
    X[bad_rows, bad_cols] = np.inf
    X[rng.integers(0, n_rows, size=n_bad), rng.integers(0, n_features, size=n_bad)] = np.nan

    df = pd.DataFrame(X, columns=feature_names)
    # Identifier columns the pipeline is expected to drop / use for grouping.
    df["Source IP"] = [f"192.168.1.{rng.integers(2, 2 + n_src_ips)}" for _ in range(n_rows)]
    df["Destination IP"] = "10.0.0.1"
    df["Timestamp"] = pd.date_range("2017-07-07", periods=n_rows, freq="s").astype(str)
    df["Label"] = [family_to_raw[f] for f in chosen]

    out_path = out_path or (config.RAW_DIR / "synthetic_cicids2017.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    path = generate_raw_csv()
    print(f"Wrote synthetic raw data -> {path}")
    print("This is shaped like CIC-IDS2017 so the real pipeline can run on it.")
