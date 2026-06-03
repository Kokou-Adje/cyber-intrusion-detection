"""
Honest evaluation of every model on the held-out test set.

WHY NOT JUST ACCURACY?
----------------------
On an 80%-benign dataset, a model that flags nothing still scores ~80%
accuracy. Accuracy hides whether attacks are actually being caught. So this
script reports the metrics a security team really cares about:

  - per-class precision / recall / F1  -> are rare attacks being detected?
  - macro averages                     -> every class counts equally, so poor
                                          attack detection can't hide behind
                                          correct benign predictions.
  - attack false-positive rate (FPR)   -> how much benign traffic is wrongly
                                          flagged (alert fatigue is the #1
                                          real-world complaint about IDS).
  - confusion matrix + PR curves       -> where exactly the model confuses classes.
  - a Random-Forest-vs-deep table      -> was the deep model worth it?

The ensemble is a SOFT vote: it averages the CNN and Transformer probabilities
(keeping confidence information) rather than counting hard label votes.

    python -m src.evaluate
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")                  # headless backend: save figures, no GUI
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_curve,
                             average_precision_score)

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
import keras


def load_test():
    """Load the held-out test split and the class-name list."""
    X = np.load(config.PROCESSED_DIR / "X_test.npy")
    y = np.load(config.PROCESSED_DIR / "y_test.npy")
    with open(config.LABEL_MAP_PATH) as f:
        label_map = json.load(f)
    # Recover names in integer order so column i == class i everywhere.
    names = [name for name, _ in sorted(label_map.items(), key=lambda kv: kv[1])]
    return X, y, names


def rf_proba(rf, X, n_classes):
    """Get Random Forest class probabilities, re-aligned to global class ids.

    A forest trained on data missing some class won't have a column for it;
    rf.classes_ tells us which columns map to which class id, so we scatter
    them into a full (n_samples, n_classes) matrix for fair comparison.
    """
    p = rf.predict_proba(X)
    full = np.zeros((len(X), n_classes), dtype=np.float32)
    for col, cls in enumerate(rf.classes_):
        full[:, int(cls)] = p[:, col]
    return full


def attack_fpr(y_true, y_pred, benign_idx):
    """Fraction of genuinely-benign flows that were flagged as some attack.

    This is the operational "noise" metric: high FPR == analysts buried in
    false alarms, even if accuracy looks great.
    """
    benign = y_true == benign_idx
    if benign.sum() == 0:
        return float("nan")
    return float((y_pred[benign] != benign_idx).sum() / benign.sum())


def summarize(name, y_true, proba, benign_idx):
    """Compute the headline metrics for one model from its probability output."""
    y_pred = proba.argmax(axis=1)                    # pick the top-probability class
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_recall": rep["macro avg"]["recall"],  # unweighted across classes
        "macro_f1": rep["macro avg"]["f1-score"],
        "weighted_f1": rep["weighted avg"]["f1-score"],
        "attack_fpr": attack_fpr(y_true, y_pred, benign_idx),
    }


def print_table(rows):
    """Render the model-comparison table as markdown for the console + README."""
    header = f"| {'Model':<22} | Accuracy | Macro Recall | Macro F1 | Weighted F1 | Attack FPR |"
    sep = "|" + "-" * 24 + "|" + "-" * 10 + "|" + "-" * 14 + "|" + "-" * 10 + "|" + "-" * 13 + "|" + "-" * 12 + "|"
    lines = [header, sep]
    for r in rows:
        lines.append(
            f"| {r['model']:<22} | {r['accuracy']:.4f}   | {r['macro_recall']:.4f}       "
            f"| {r['macro_f1']:.4f}   | {r['weighted_f1']:.4f}      | {r['attack_fpr']:.4f}     |"
        )
    table = "\n".join(lines)
    print("\n" + table + "\n")
    return table


def plot_confusion(y_true, y_pred, names, path):
    """Row-normalized confusion matrix: each row shows where a true class went."""
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Ensemble confusion matrix (row-normalized)")
    for i in range(len(names)):                      # annotate each cell
        for j in range(len(names)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_pr_curves(y_true, proba, names, path):
    """One-vs-rest precision-recall curve per class.

    PR curves are more informative than ROC on imbalanced data because they
    focus on the positive (attack) class instead of being dominated by the
    huge benign majority.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    for cls, name in enumerate(names):
        binary = (y_true == cls).astype(int)
        if binary.sum() == 0:
            continue
        precision, recall, _ = precision_recall_curve(binary, proba[:, cls])
        ap = average_precision_score(binary, proba[:, cls])   # area under PR
        ax.plot(recall, precision, label=f"{name} (AP={ap:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Ensemble precision-recall (one-vs-rest)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    X_test, y_test, names = load_test()
    n_classes = len(names)
    # Index of the benign class, needed by the FPR metric. Default to 0 if a
    # dataset somehow has no benign class.
    benign_idx = names.index("Benign") if "Benign" in names else 0

    # Load all three trained models.
    rf = joblib.load(config.MODELS_DIR / "rf_baseline.pkl")
    cnn = keras.models.load_model(config.CNN_PATH)
    trans = keras.models.load_model(config.TRANSFORMER_PATH)

    # Collect each model's probability output on the test set.
    proba = {
        "Random Forest": rf_proba(rf, X_test, n_classes),
        "CNN": cnn.predict(X_test, verbose=0),
        "Transformer": trans.predict(X_test, verbose=0),
    }
    # SOFT VOTE: average the two deep models' probabilities.
    proba["Ensemble (soft vote)"] = (proba["CNN"] + proba["Transformer"]) / 2.0

    # Build and print the comparison table.
    rows = [summarize(name, y_test, p, benign_idx) for name, p in proba.items()]
    table = print_table(rows)

    # Detailed per-class report for the ensemble (the model we ship).
    ens_pred = proba["Ensemble (soft vote)"].argmax(axis=1)
    print("Per-class report (ensemble):")
    print(classification_report(y_test, ens_pred, target_names=names, zero_division=0))

    # Save figures + machine-readable metrics for the README.
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion(y_test, ens_pred, names, config.REPORTS_DIR / "confusion_matrix.png")
    plot_pr_curves(y_test, proba["Ensemble (soft vote)"], names,
                   config.REPORTS_DIR / "pr_curves.png")
    with open(config.REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(rows, f, indent=2)
    with open(config.REPORTS_DIR / "results.md", "w") as f:
        f.write("# Results\n\n" + table + "\n")
    print(f"[done] figures + metrics saved to {config.REPORTS_DIR}")


if __name__ == "__main__":
    main()
