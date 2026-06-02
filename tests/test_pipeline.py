"""
Unit tests covering the parts most likely to break silently.

Run with:  pytest -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src import preprocess as pp


def _tiny_df(n=300, feats=config.NUM_FEATURES, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, feats))
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(feats)])
    df["Source IP"] = [f"10.0.0.{i % 5}" for i in range(n)]
    df["Timestamp"] = pd.date_range("2017-01-01", periods=n, freq="s").astype(str)
    df["Label"] = rng.choice(["BENIGN", "DDoS"], size=n)
    return df


def test_sequence_shape():
    df = _tiny_df()
    feats, y, src, ts = pp.select_features(df)
    labels, _ = pp.encode_labels(y)
    X, yy = pp.build_sequences(feats, labels, df[src], df[ts], config.SEQ_LEN)
    assert X.shape[1:] == (config.SEQ_LEN, feats.shape[1])
    assert len(X) == len(yy)


def test_scaler_fit_on_train_only():
    """Training data must land in [0,1]; the scaler must be a MinMaxScaler.
    This guards against the classic leakage bug (fitting on the full set)."""
    from sklearn.preprocessing import MinMaxScaler
    df = _tiny_df()
    feats, y, src, ts = pp.select_features(df)
    labels, _ = pp.encode_labels(y)
    X, yy = pp.build_sequences(feats, labels, df[src], df[ts], config.SEQ_LEN)
    (Xtr, _), (Xva, _), (Xte, _), scaler = pp.split_and_scale(X, yy)
    assert isinstance(scaler, MinMaxScaler)
    assert Xtr.min() >= -1e-6 and Xtr.max() <= 1 + 1e-6


def test_identifier_columns_dropped():
    """No IP/timestamp leaks into the feature matrix."""
    df = _tiny_df()
    feats, _, _, _ = pp.select_features(df)
    leaked = {c for c in feats.columns if "ip" in c.lower() or "time" in c.lower()}
    assert leaked == set()


def test_model_output_shapes():
    from src.models import build_cnn, build_transformer
    cnn = build_cnn(n_classes=3)
    trans = build_transformer(n_classes=3)
    batch = np.random.rand(2, config.SEQ_LEN, config.NUM_FEATURES).astype("float32")
    assert cnn.predict(batch, verbose=0).shape == (2, 3)
    assert trans.predict(batch, verbose=0).shape == (2, 3)


def test_attack_fpr():
    from src.evaluate import attack_fpr
    y_true = np.array([0, 0, 1, 1])      # benign = class 0
    y_pred = np.array([0, 1, 1, 1])      # one benign wrongly flagged
    assert attack_fpr(y_true, y_pred, benign_idx=0) == pytest.approx(0.5)
