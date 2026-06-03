"""
Streamlit dashboard for the per-flow intrusion detection ensemble.

Loads the trained CNN + Transformer + fitted scaler and classifies a single
network flow, showing the alert decision, predicted class, confidence, and the
full probability breakdown. Three input modes: sample from the held-out test
set, upload a CSV, or generate a random flow.

    streamlit run app/intrusion_dashboard.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
import keras

st.set_page_config(page_title="Intrusion Detection", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_artifacts():
    """Load models, scaler, and label names once and cache them."""
    missing = [p for p in (config.CNN_PATH, config.TRANSFORMER_PATH,
                           config.SCALER_PATH, config.LABEL_MAP_PATH)
               if not Path(p).exists()]
    if missing:
        return None
    cnn = keras.models.load_model(config.CNN_PATH)
    trans = keras.models.load_model(config.TRANSFORMER_PATH)
    scaler = joblib.load(config.SCALER_PATH)
    with open(config.LABEL_MAP_PATH) as f:
        label_map = json.load(f)
    names = [n for n, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    return cnn, trans, scaler, names


def predict(cnn, trans, names, flow_scaled):
    """Soft-vote the two models. flow_scaled: (1, NUM_FEATURES)."""
    p = (cnn.predict(flow_scaled, verbose=0) + trans.predict(flow_scaled, verbose=0)) / 2.0
    p = p[0]
    idx = int(p.argmax())
    return names[idx], float(p[idx]), p


st.title("🛡️ Network Intrusion Detection")
st.caption(f"CNN + Transformer soft-voting ensemble, classifying one network "
           f"flow of {config.NUM_FEATURES} features at a time.")

artifacts = load_artifacts()
if artifacts is None:
    st.warning("Models not found. Run the pipeline first:\n\n"
               "```\npython -m src.preprocess\npython -m src.train\n```")
    st.stop()

cnn, trans, scaler, names = artifacts

with st.sidebar:
    st.header("Input")
    mode = st.radio("Flow source",
                    ["Sample from test set", "Upload CSV", "Random flow"])

if mode == "Sample from test set":
    test_path = config.PROCESSED_DIR / "X_test.npy"
    if not test_path.exists():
        st.error("No processed test set found. Run `python -m src.preprocess`.")
        st.stop()
    X_test = np.load(test_path)
    y_test = np.load(config.PROCESSED_DIR / "y_test.npy")
    i = st.sidebar.slider("Test flow index", 0, len(X_test) - 1, 0)
    flow_scaled = X_test[i][None, :]                 # already scaled
    st.sidebar.info(f"True label: **{names[int(y_test[i])]}**")
elif mode == "Upload CSV":
    up = st.sidebar.file_uploader(
        f"CSV with {config.NUM_FEATURES} numeric feature columns", type="csv")
    if up is None:
        st.info("Upload a CSV to score its first row.")
        st.stop()
    df = pd.read_csv(up).select_dtypes(include=[np.number])
    if df.shape[1] != config.NUM_FEATURES:
        st.error(f"Need {config.NUM_FEATURES} numeric columns. Got {df.shape[1]}.")
        st.stop()
    flow = df.to_numpy()[0:1].astype("float32")      # score the first row
    flow_scaled = scaler.transform(flow)
else:
    rng = np.random.default_rng()
    flow = rng.random((1, config.NUM_FEATURES)).astype("float32")
    flow_scaled = scaler.transform(flow)

label, confidence, proba = predict(cnn, trans, names, flow_scaled)
is_attack = label != "Benign"

c1, c2 = st.columns(2)
with c1:
    if is_attack:
        st.error(f"### 🚨 ALERT: {label}")
    else:
        st.success("### ✅ Benign traffic")
with c2:
    st.metric("Confidence", f"{confidence:.1%}")

st.subheader("Class probabilities")
st.bar_chart(pd.DataFrame({"probability": proba}, index=names))

with st.expander("Show the scaled input flow"):
    st.dataframe(pd.DataFrame(flow_scaled,
                              columns=[f"f{i}" for i in range(config.NUM_FEATURES)]))
