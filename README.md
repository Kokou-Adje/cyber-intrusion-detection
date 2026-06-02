# 🛡️ Network Intrusion Detection

> Detecting network attacks with a CNN + Transformer ensemble on CIC-IDS2017
> CS 7357: Neural Networks and Deep Learning — Fall 2025
> Kennesaw State University

![Python](https://img.shields.io/badge/Python-3.10+-3776AB)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17-FF6F00)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30-FF4B4B)

## 🎯 Project overview

Detecting network attacks from flow records using a deep learning ensemble,
benchmarked honestly against a strong classical baseline.

The model classifies short sequences of network flows (15 consecutive flows ×
78 features) into traffic categories: benign, DDoS, DoS, probe, and web attack.
A 1D-CNN and a Transformer each make a prediction, and their probabilities are
averaged into a soft-voting ensemble.

![ML pipeline data flow](assets/pipeline.svg)

> **Status / honesty note:** the pipeline runs end to end on a synthetic,
> CIC-IDS2017-shaped dataset so you can clone and run it in five minutes. The
> headline metrics in `reports/` come from that synthetic data and are *not*
> real performance numbers — synthetic classes are cleanly separable, so
> everything scores near-perfect. To get real results, drop the actual
> CIC-IDS2017 CSVs into `data/raw/` and rerun. The whole point of the design is
> that nothing else changes when you do.

## 🧠 Why this project is built the way it is

Most intrusion-detection demos report 99% accuracy and stop there. That number
is misleading: the dataset is roughly 80% benign, so a model that flags nothing
still looks "80% accurate." This project is built around the questions a
security team actually asks.

**Are rare attacks being caught?** Evaluation reports per-class recall and
macro-averaged F1, not just overall accuracy, so a model can't hide poor
attack detection behind a pile of correct benign predictions.

**How noisy are the alerts?** A dedicated attack false-positive-rate metric
measures how often benign traffic is wrongly flagged. Alert fatigue is the
number-one complaint about real intrusion detection systems, so it gets its own
column.

**Is the deep model even worth it?** A Random Forest baseline is trained
first, and the deep models have to beat it. On tabular network features, tree
ensembles are genuinely strong — sometimes they win. The project lets the
evidence decide instead of assuming the fancy model is better.

**No data leakage.** Identifier columns (IP addresses, ports, timestamps) are
dropped before training so the model learns behavior, not which machine sent
the traffic. The scaler is fit on the training split only, then applied to
validation and test — fitting on everything is the most common silent mistake
in student ML projects, and it inflates every number.

## 🏗️ Architecture

![Model ensemble structure](assets/ensemble.svg)

The CNN stacks two 1D-convolution layers (local burst patterns). The
Transformer projects the features, adds sinusoidal positional encoding, and
runs two self-attention encoder blocks (long-range patterns across the window).

## 🚀 Quickstart

```bash
pip install -r requirements.txt

# 1. generate synthetic CIC-IDS2017-shaped data (or skip and use real CSVs)
python -m src.synthetic_data

# 2. preprocess: clean, sequence, split, fit + save the scaler
python -m src.preprocess

# 3. train the baseline + CNN + Transformer
python -m src.train            # add --epochs 4 for a quick smoke test

# 4. evaluate everything on the held-out test set
python -m src.evaluate

# 5. launch the demo
streamlit run app/intrusion_dashboard.py
```

Tests: `pytest -q`

## 📊 Using the real dataset

Download CIC-IDS2017 (Canadian Institute for Cybersecurity), put the CSVs in
`data/raw/`, delete the synthetic file, and rerun from step 2. If the real data
yields a different feature count after dropping identifiers, update
`NUM_FEATURES` in `config.py` (the preprocessing script prints the count it
ends with).

## 📁 Repository layout

```
config.py                  all paths, shapes, and hyperparameters
src/synthetic_data.py      generates CIC-IDS2017-shaped test data
src/preprocess.py          clean → select → sequence → split → scale
src/models.py              1D-CNN and Transformer definitions
src/train.py               baseline + class-weighted deep training
src/evaluate.py            metrics, confusion matrix, PR curves, comparison
app/intrusion_dashboard.py Streamlit inference demo
tests/test_pipeline.py     leakage, shape, and output-distribution tests
reports/                   generated metrics and figures
assets/                    README diagrams (pipeline.svg, ensemble.svg)
docs/PIPELINE_REFERENCE.md step-by-step map of pipeline to code (interview prep)
```

## ⚠️ Limitations

This is a research/portfolio project, not a production IDS. Worth knowing:

- **CIC-IDS2017 is from 2017.** Attack techniques and normal traffic have both
  moved on. A model trained here would need retraining and validation on
  current traffic before it meant anything operationally.
- **Flow features only.** It works on statistical flow summaries, not packet
  payloads, so it can't inspect encrypted-traffic contents.
- **Concept drift is unhandled.** Real deployments need monitoring and periodic
  retraining as traffic patterns shift; none of that is here.
- **Synthetic default data is easy.** The bundled generator produces cleanly
  separable classes for testability. Treat the out-of-the-box metrics as a
  smoke test, not a result.

## 📄 License

MIT — see `LICENSE`.
