"""
Model architectures for per-flow intrusion classification.

Each model classifies a SINGLE network flow (a vector of NUM_FEATURES numbers)
into a traffic category. Three models are compared:

  build_cnn          - reshapes the flow to (features, 1) and slides 1D
                       convolutions across the feature axis to learn local
                       feature interactions.
  build_transformer  - treats the features as a short token sequence and uses
                       self-attention to learn relationships between features.
  (Random Forest baseline lives in src/train.py.)

The CNN and Transformer are averaged into a soft-vote ensemble in evaluate.py.
"""
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf  # noqa: F401  (importing confirms the TF backend loads)
import keras
from keras import layers

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def build_cnn(n_features: int = config.NUM_FEATURES,
              n_classes: int = 2,
              lr: float = config.LEARNING_RATE) -> keras.Model:
    """1D-CNN over the feature vector. Reshaping to (n_features, 1) lets the
    convolution slide across features and learn local groups of related
    features (e.g. the packet-length stats that move together)."""
    inputs = keras.Input(shape=(n_features,))
    x = layers.Reshape((n_features, 1))(inputs)          # (features, 1 channel)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)                   # stabilizes training
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)                        # keep strongest signals
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)                           # regularization
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="cnn_1d")
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy",   # integer labels
                  metrics=["accuracy"])
    return model


def _encoder_block(x, d_model: int, num_heads: int, ff_dim: int, dropout: float):
    """One Transformer encoder block: self-attention + feed-forward, each with a
    residual connection and layer norm."""
    attn = layers.MultiHeadAttention(num_heads=num_heads,
                                     key_dim=d_model // num_heads,
                                     dropout=dropout)(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn)
    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff = layers.Dense(d_model)(ff)
    ff = layers.Dropout(dropout)(ff)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ff)
    return x


def build_transformer(n_features: int = config.NUM_FEATURES,
                      n_classes: int = 2,
                      d_model: int = 64,             # per-feature embedding width
                      num_heads: int = 4,
                      ff_dim: int = 128,
                      num_blocks: int = 2,
                      dropout: float = 0.2,
                      lr: float = config.LEARNING_RATE) -> keras.Model:
    """Transformer that treats each feature as a token and attends across them.
    No positional encoding here: feature order in a flow is arbitrary (unlike a
    time sequence), so we deliberately let attention be order-independent."""
    inputs = keras.Input(shape=(n_features,))
    # Treat the n_features values as a sequence of n_features tokens, each a
    # scalar projected up to d_model dimensions.
    x = layers.Reshape((n_features, 1))(inputs)
    x = layers.Dense(d_model)(x)                     # (n_features, d_model)
    for _ in range(num_blocks):
        x = _encoder_block(x, d_model, num_heads, ff_dim, dropout)
    x = layers.GlobalAveragePooling1D()(x)           # pool across feature-tokens
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="transformer")
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


if __name__ == "__main__":
    cnn = build_cnn(n_classes=9)
    trans = build_transformer(n_classes=9)
    print(f"CNN params:         {cnn.count_params():,}")
    print(f"Transformer params: {trans.count_params():,}")
    batch = np.random.rand(4, config.NUM_FEATURES).astype("float32")
    print("CNN output shape:        ", cnn.predict(batch, verbose=0).shape)
    print("Transformer output shape:", trans.predict(batch, verbose=0).shape)
