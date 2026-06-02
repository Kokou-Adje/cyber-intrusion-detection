"""
Model architectures for sequence-based intrusion detection.

WHY TWO DEEP MODELS?
--------------------
They look at the same (SEQ_LEN, NUM_FEATURES) sequence in complementary ways:

  build_cnn          - 1D convolutions slide small filters across consecutive
                       flows, so they capture SHORT, LOCAL patterns (e.g. a
                       sudden burst of similar packets a few flows wide).
  build_transformer  - self-attention lets every flow look at every other flow
                       in the window, so it captures LONG-RANGE relationships
                       (e.g. reconnaissance early, exfiltration later).

Because they make different kinds of mistakes, averaging them (the soft-vote
ensemble in evaluate.py) is usually more robust than either alone. The tree
baseline they must beat lives in src/train.py.
"""
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf  # noqa: F401  (importing confirms the TF backend loads)
import keras
from keras import layers

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


# Registering the layer makes Keras able to reload a saved .keras model that
# uses it WITHOUT us passing custom_objects every time. Without this decorator,
# `keras.models.load_model(...)` on the Transformer would raise an "unknown
# layer" error - a classic reason a portfolio repo looks untested when someone
# tries to run it.
@keras.saving.register_keras_serializable(package="ids")
class PositionalEncoding(layers.Layer):
    """Add fixed sinusoidal positional information to each flow in the window.

    Self-attention is permutation-invariant: on its own it cannot tell flow #1
    from flow #15, so it would treat a sequence and its shuffle identically.
    Injecting a position signal restores order, which matters because attack
    patterns are temporal.
    """

    def __init__(self, seq_len: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.d_model = d_model
        # Precompute the encoding once; it is constant (not learned).
        self.pos_encoding = self._build_encoding(seq_len, d_model)

    @staticmethod
    def _build_encoding(seq_len: int, d_model: int) -> tf.Tensor:
        # Standard "Attention Is All You Need" sinusoidal scheme: each position
        # gets a unique combination of sine/cosine waves at different
        # frequencies, so the model can infer relative distances between flows.
        pos = np.arange(seq_len)[:, None]
        i = np.arange(d_model)[None, :]
        angle_rates = 1.0 / np.power(10000.0, (2 * (i // 2)) / np.float32(d_model))
        angles = pos * angle_rates
        angles[:, 0::2] = np.sin(angles[:, 0::2])     # even dims -> sine
        angles[:, 1::2] = np.cos(angles[:, 1::2])     # odd dims  -> cosine
        return tf.constant(angles[None, ...], dtype=tf.float32)

    def call(self, x):
        # Broadcast-add the position signal onto the projected input.
        return x + self.pos_encoding

    def get_config(self):
        # Needed so the layer's constructor args survive save/load.
        cfg = super().get_config()
        cfg.update({"seq_len": self.seq_len, "d_model": self.d_model})
        return cfg


def build_cnn(seq_len: int = config.SEQ_LEN,
              n_features: int = config.NUM_FEATURES,
              n_classes: int = 2,
              lr: float = config.LEARNING_RATE) -> keras.Model:
    """1D-CNN for local temporal patterns across a few consecutive flows."""
    inputs = keras.Input(shape=(seq_len, n_features))
    # Two stacked Conv1D layers learn increasingly complex local patterns.
    # padding="same" keeps the time dimension intact so we don't lose flows.
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)               # stabilizes + speeds training
    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)                    # downsample: keep strongest signals
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)                       # regularization: fights overfitting
    # Softmax over classes -> a probability distribution that sums to 1.
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="cnn_1d")
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  # sparse_* lets labels stay as integers (0,1,2,...) instead of
                  # one-hot vectors - less memory, same result.
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def _encoder_block(x, d_model: int, num_heads: int, ff_dim: int, dropout: float):
    """One Transformer encoder block: attention + feed-forward, each with a
    residual connection and layer norm. Residuals let gradients flow through
    deep stacks; layer norm keeps activations well-scaled."""
    # --- Sub-layer 1: multi-head self-attention --------------------------------
    attn = layers.MultiHeadAttention(num_heads=num_heads,
                                     key_dim=d_model // num_heads,
                                     dropout=dropout)(x, x)   # query=key=value=x
    x = layers.LayerNormalization(epsilon=1e-6)(x + attn)     # residual + norm
    # --- Sub-layer 2: position-wise feed-forward -------------------------------
    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff = layers.Dense(d_model)(ff)
    ff = layers.Dropout(dropout)(ff)
    x = layers.LayerNormalization(epsilon=1e-6)(x + ff)       # residual + norm
    return x


def build_transformer(seq_len: int = config.SEQ_LEN,
                      n_features: int = config.NUM_FEATURES,
                      n_classes: int = 2,
                      d_model: int = 128,            # internal embedding width
                      num_heads: int = 4,            # parallel attention "views"
                      ff_dim: int = 256,             # feed-forward hidden size
                      num_blocks: int = 2,           # stacked encoder blocks
                      dropout: float = 0.2,
                      lr: float = config.LEARNING_RATE) -> keras.Model:
    """Transformer encoder for long-range patterns across the whole window."""
    inputs = keras.Input(shape=(seq_len, n_features))
    x = layers.Dense(d_model)(inputs)                # project 78 features -> d_model
    x = PositionalEncoding(seq_len, d_model)(x)      # inject flow order
    for _ in range(num_blocks):                      # stack encoder blocks
        x = _encoder_block(x, d_model, num_heads, ff_dim, dropout)
    # Collapse the time dimension into one vector before classifying.
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="transformer")
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


if __name__ == "__main__":
    # Quick smoke test: build both models and run one forward pass each.
    cnn = build_cnn(n_classes=5)
    trans = build_transformer(n_classes=5)
    print(f"CNN params:         {cnn.count_params():,}")
    print(f"Transformer params: {trans.count_params():,}")
    batch = np.random.rand(4, config.SEQ_LEN, config.NUM_FEATURES).astype("float32")
    print("CNN output shape:        ", cnn.predict(batch, verbose=0).shape)
    print("Transformer output shape:", trans.predict(batch, verbose=0).shape)
