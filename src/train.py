"""
Train the baseline and both deep models.

WHY A BASELINE FIRST?
---------------------
The single most credible thing in this project is that the deep models have to
beat a classical one. On tabular network features, tree ensembles are famously
strong - sometimes they win. Training a Random Forest first turns "is the CNN +
Transformer worth the extra complexity?" into a question the results answer,
instead of an assumption. Order: (1) Random Forest, (2) 1D-CNN, (3) Transformer.

WHY CLASS WEIGHTS?
------------------
CIC-IDS2017 is ~80% benign. With no correction, the cheapest way for a model to
minimize loss is to predict "benign" for everything and ignore rare attacks.
Class weights make the loss penalize a missed rare-class sample far more than a
missed benign one, so the model is pushed to actually detect attacks.

    python -m src.train                  # full run (config.EPOCHS)
    python -m src.train --epochs 3        # quick smoke test
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
import keras
from keras import callbacks
# Importing the models module also registers the custom PositionalEncoding layer.
from src.models import build_cnn, build_transformer


def load_split(name: str):
    """Load a saved (X, y) split produced by preprocess.py."""
    X = np.load(config.PROCESSED_DIR / f"X_{name}.npy")
    y = np.load(config.PROCESSED_DIR / f"y_{name}.npy")
    return X, y


def class_weight_dict(y: np.ndarray) -> dict[int, float]:
    """Compute inverse-frequency weights so rare classes count more.

    'balanced' sets weight = n_samples / (n_classes * count_of_class), i.e.
    common classes get weight < 1, rare classes get weight > 1.
    """
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def train_baseline(X_train, y_train):
    """Train the Random Forest the deep models must beat."""
    print("\n=== Random Forest baseline ===")
    t0 = time.time()
    # Trees need 2D input, so flatten each (seq_len, features) window into one
    # long feature vector. class_weight="balanced" applies the same imbalance
    # correction the deep models get via class weights.
    rf = RandomForestClassifier(
        n_estimators=200, n_jobs=-1, class_weight="balanced",
        random_state=config.RANDOM_SEED,            # reproducible
    )
    rf.fit(X_train.reshape(len(X_train), -1), y_train)
    joblib.dump(rf, config.MODELS_DIR / "rf_baseline.pkl")
    print(f"trained in {time.time() - t0:.1f}s -> models/rf_baseline.pkl")
    return rf


def train_keras(model, X_train, y_train, X_val, y_val, cw, epochs, save_path):
    """Train a Keras model with early stopping and LR reduction."""
    print(f"\n=== {model.name} ===")
    cbs = [
        # Stop when validation loss stops improving and restore the best
        # weights - prevents overfitting and wasted epochs.
        callbacks.EarlyStopping(monitor="val_loss",
                                patience=config.EARLY_STOPPING_PATIENCE,
                                restore_best_weights=True),
        # If val loss plateaus, halve the learning rate to fine-tune.
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]
    history = model.fit(
        X_train, y_train, validation_data=(X_val, y_val),
        epochs=epochs, batch_size=config.BATCH_SIZE,
        class_weight=cw,                              # <-- imbalance correction
        callbacks=cbs, verbose=2,
    )
    model.save(save_path)                             # persist the trained model
    print(f"saved -> {save_path}")
    return history.history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    args = parser.parse_args()

    # One call seeds Python, NumPy, and TensorFlow RNGs -> reproducible runs.
    keras.utils.set_random_seed(config.RANDOM_SEED)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    # n_classes from the data, not hard-coded, so the heads size themselves.
    n_classes = int(np.load(config.PROCESSED_DIR / "y_train.npy").max()) + 1
    n_features = X_train.shape[-1]
    cw = class_weight_dict(y_train)
    print(f"classes={n_classes} features={n_features} class_weights={cw}")

    # 1) baseline
    train_baseline(X_train, y_train)

    # 2) CNN
    histories = {}
    cnn = build_cnn(n_features=n_features, n_classes=n_classes)
    histories["cnn"] = train_keras(cnn, X_train, y_train, X_val, y_val,
                                   cw, args.epochs, config.CNN_PATH)

    # 3) Transformer
    trans = build_transformer(n_features=n_features, n_classes=n_classes)
    histories["transformer"] = train_keras(trans, X_train, y_train, X_val, y_val,
                                           cw, args.epochs, config.TRANSFORMER_PATH)

    # Save learning curves so they can be plotted / discussed later.
    with open(config.REPORTS_DIR / "training_history.json", "w") as f:
        json.dump({k: {m: [float(v) for v in vals] for m, vals in h.items()}
                   for k, h in histories.items()}, f, indent=2)
    print("\n[done] all three models trained. Run `python -m src.evaluate`.")


if __name__ == "__main__":
    main()
