# Pipeline reference (interview prep)

Each ML pipeline step mapped to the exact file, function, and line where it
happens, plus the one-sentence "why" to say out loud. Line numbers match the
commented source in this repo.

## The 15 steps

| # | Step | Where | Why (say this) |
|---|------|-------|----------------|
| 1 | Problem framing | — (README intro) | Multi-class classification of flow sequences; goal is catching attacks without flooding analysts, which drives the metric choices. |
| 2 | Data ingestion | `src/preprocess.py` → `load_raw` (L58) | Merge all CSVs and strip the leading-space headers CIC-IDS2017 ships with, so later column lookups don't crash. |
| 3 | Data cleaning | `src/preprocess.py` → `clean` (L80) | Convert Inf→NaN, drop NaN, drop duplicate flows — Inf explodes gradients and duplicates leak across splits. |
| 4 | Feature selection / leakage prevention | `src/preprocess.py` → `select_features` (L98) | Drop identifiers (IP, port, timestamp) so the model learns behavior, not which host sent the traffic. |
| 5 | Target preparation | `src/preprocess.py` → `encode_labels` (L131) | Collapse ~15 raw labels into coarse families and integer-encode; rare raw classes are too small to learn. |
| 6 | Sequencing | `src/preprocess.py` → `build_sequences` (L152) | Slide a 15-flow window grouped by source IP; an attack is a pattern across flows, not one flow. |
| 7 | Train/val/test split | `src/preprocess.py` → `split_and_scale` (L204) | Stratified split *before* scaling so rare classes appear in every split and nothing leaks. |
| 8 | Scaling | `src/preprocess.py` → `split_and_scale` (L204, `scaler.fit` on train) | Fit MinMax on training flows only; fitting on all data leaks test statistics and inflates results. |
| 9 | Artifact persistence | `src/preprocess.py` → `main` (L245, `np.save`/`joblib.dump`) | Save arrays, scaler, label map so training and the demo reuse identical, reproducible inputs. |
| 10 | Model building | `src/models.py` → `build_cnn` (L77), `build_transformer` (L122); baseline in `train.py` → `train_baseline` (L59) | CNN = local patterns, Transformer = long-range patterns, Random Forest = the bar to beat. |
| 11 | Training + imbalance handling | `src/train.py` → `train_keras` (L76), weights from `class_weight_dict` (L48) | Class weights stop the model defaulting to "benign"; early stopping + LR reduction prevent overfitting. |
| 12 | Evaluation | `src/evaluate.py` → `summarize` (L81), `attack_fpr` (L69), `plot_confusion` (L110), `plot_pr_curves` (L130) | Per-class recall, macro F1, and attack false-positive rate — accuracy alone hides missed attacks. |
| 13 | Ensembling | `src/evaluate.py` → `main` (L154, soft-vote average) | Average CNN + Transformer probabilities; two architectures fail differently, so averaging is more robust. |
| 14 | Deployment / serving | `app/intrusion_dashboard.py` → `load_artifacts` (L29), `predict` (L45) | Reuse the saved scaler + models so live input is processed exactly as training was. |
| 15 | Testing + reproducibility | `tests/test_pipeline.py` (L28–L73), `config.py` (seed L64) | Tests guard leakage, shapes, and valid probabilities; fixed seed + central config make runs repeatable. |

## Config values worth knowing cold

| Value | Location | Meaning |
|-------|----------|---------|
| `SEQ_LEN = 15` | `config.py` L22 | Flows per sequence window |
| `NUM_FEATURES = 78` | `config.py` L23 | Features kept after dropping identifiers |
| `LABEL_GROUPING` | `config.py` L29 | Raw-label → family map |
| `IDENTIFIER_COLUMNS` | `config.py` L50 | Columns dropped to prevent leakage |
| `TEST_SIZE / VAL_SIZE = 0.15` | `config.py` L62–63 | Split fractions |
| `RANDOM_SEED = 42` | `config.py` L64 | Reproducibility |
| `EPOCHS = 30`, `BATCH_SIZE = 256` | `config.py` L66–67 | Training schedule |

## The three questions an interviewer will probe

1. **"How do you know you're not leaking?"** → Two answers: identifiers are
   dropped in `select_features` (L98), and the scaler is fit on train only in
   `split_and_scale` (L204). A unit test (`test_scaler_fit_on_train_only`, L37)
   enforces the second.

2. **"Why a Transformer and a CNN, not just one?"** → They capture different
   structure (local vs long-range), make different errors, and the soft-vote
   ensemble averages them for robustness. The baseline exists to check the deep
   models are even worth it.

3. **"Why isn't accuracy your headline metric?"** → The dataset is ~80% benign,
   so accuracy rewards predicting "benign" for everything. Macro recall and
   attack false-positive rate measure what actually matters operationally.
