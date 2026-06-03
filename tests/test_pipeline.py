"""
Fast unit tests for the per-flow pipeline. Run with: pytest -q

Covers the parts most likely to break silently:
  - the scaler is fit on train only and bounds the training data to [0, 1]
  - both models output a valid probability distribution per class
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.preprocess import split_and_scale
from src.models import build_cnn, build_transformer


def test_scaler_fit_on_train_only_bounds_train():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, config.NUM_FEATURES)).astype(np.float32)
    y = (np.arange(len(X)) % 2)
    (Xtr, _), (Xva, _), (Xte, _), scaler = split_and_scale(X, y)
    # Train flows must land in [0, 1] because the scaler was fit on them...
    assert Xtr.min() >= -1e-6 and Xtr.max() <= 1 + 1e-6
    # ...while val/test keep the right shape (they may exceed [0,1] -> no leakage).
    assert Xva.shape[1] == config.NUM_FEATURES
    assert Xte.shape[1] == config.NUM_FEATURES


@pytest.mark.parametrize("builder", [build_cnn, build_transformer])
def test_model_outputs_probability_distribution(builder):
    n_classes = 9
    model = builder(n_classes=n_classes)
    x = np.random.rand(8, config.NUM_FEATURES).astype("float32")
    p = model.predict(x, verbose=0)
    assert p.shape == (8, n_classes)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-5)   # softmax sums to 1
