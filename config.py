"""
Central configuration for the Cyber Intrusion Detection pipeline.

Every script imports from here so paths, shapes, and hyperparameters live in
one place. Change a value here and the whole pipeline follows.
"""
from pathlib import Path

# --- Paths (absolute, derived from this file's location) -------------------
ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"          # put CIC-IDS2017 CSVs here
PROCESSED_DIR = ROOT / "data" / "processed"  # .npy arrays land here
MODELS_DIR = ROOT / "models"             # saved .keras models + scaler.pkl
REPORTS_DIR = ROOT / "reports"           # metrics + figures

SCALER_PATH = MODELS_DIR / "scaler.pkl"
LABEL_MAP_PATH = MODELS_DIR / "label_map.json"
CNN_PATH = MODELS_DIR / "cnn_model.keras"
TRANSFORMER_PATH = MODELS_DIR / "transformer_model.keras"

# --- Feature shape ----------------------------------------------------------
# CIC-IDS2017's machine-learning CSVs classify ONE network flow at a time
# (they carry no source-IP/timestamp context, so time-ordered sequences aren't
# meaningful on this data - per-flow is the standard, honest approach here).
NUM_FEATURES = 76   # numeric features kept after dropping identifiers + label

# --- Label handling ---------------------------------------------------------
# CIC-IDS2017 has ~15 raw labels. We collapse them into coarse families so the
# task is learnable and the classes are interpretable. Edit this mapping to
# change the granularity of the problem. Anything not listed -> "Other".
LABEL_GROUPING = {
    "BENIGN": "Benign",
    "DDoS": "DDoS",
    "DoS Hulk": "DoS",
    "DoS GoldenEye": "DoS",
    "DoS slowloris": "DoS",
    "DoS Slowhttptest": "DoS",
    "PortScan": "Probe",
    "FTP-Patator": "BruteForce",
    "SSH-Patator": "BruteForce",
    "Web Attack \u2013 Brute Force": "WebAttack",
    "Web Attack \u2013 XSS": "WebAttack",
    "Web Attack \u2013 Sql Injection": "WebAttack",
    "Bot": "Botnet",
    "Infiltration": "Infiltration",
    "Heartbleed": "Heartbleed",
}

# Columns that identify a flow rather than describe its behavior. Keeping these
# would let the model "cheat" (e.g., memorize attacker IPs), so we drop them.
# Names are matched case-insensitively after whitespace stripping.
IDENTIFIER_COLUMNS = [
    "Flow ID", "Source IP", "Src IP", "Destination IP", "Dst IP",
    "Source Port", "Src Port", "Destination Port", "Dst Port",
    "Protocol", "Timestamp", "Fwd Header Length.1",
]

# Column names used to order and group flows into sequences, if present.
SRC_IP_CANDIDATES = ["Source IP", "Src IP"]
TIMESTAMP_CANDIDATES = ["Timestamp"]
LABEL_CANDIDATES = ["Label"]

# --- Splits / training ------------------------------------------------------
TEST_SIZE = 0.15
VAL_SIZE = 0.15      # fraction of the full dataset (taken from the train part)
RANDOM_SEED = 42

EPOCHS = 30
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
EARLY_STOPPING_PATIENCE = 5
