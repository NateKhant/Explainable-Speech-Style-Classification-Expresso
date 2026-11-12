# Expresso: eGeMAPS → logistic regression (baseline path)

Glass-box baseline for the Black Box group's Explainable Speech Style Classification project.
The write-up of training, results and the explanation mechanism is in `report/Report.pdf`;
this README covers setup, output files and how to reproduce everything.

## Setup
```
cd "Expresso Project"
python3.12 -m venv .venv            # or: uv venv --seed --python 3.12 .venv
.venv/bin/python -m pip install -r requirements.txt
```
- Runs on CPU; no GPU needed. Tested on an Apple M5 (16 GB RAM) with Python 3.12.
- No ffmpeg or torch needed: audio is read with `soundfile`.
- All package versions are pinned in `requirements.txt` (key ones: datasets 5.1.0, soundfile 0.14.0,
  opensmile 2.6.0, scikit-learn 1.9.1, pandas 3.1.1, numpy 2.5.4, scipy 1.18.1, matplotlib 3.11.2).
- Don't rename this folder after creating `.venv`: the environment stores its own path.

## Reproduce from scratch
Run from this folder, in order:

| Step | Command | Output | Time (Apple M5) |
|---|---|---|---|
| 0 Download | `.venv/bin/python scripts/00_download_expresso.py` | `.hf_cache/` (5.4 GB), `clips.csv` | download-bound |
| 1 Features | `.venv/bin/python scripts/01_extract_egemaps.py` | `features/egemaps.parquet` | ~1 min |
| 2 Folds | `.venv/bin/python scripts/02_make_folds.py` | `splits/folds.csv` | seconds |
| 3 Train | `.venv/bin/python scripts/03_train_logreg.py` | `models/`, most of `results/` | ~10 s |
| 4 Explanations | `.venv/bin/python scripts/04_group_contributions.py` | `results/feature_groups.csv`, `results/group_contributions_*.csv` | seconds |
| 5 Report | `python scripts/05_build_report.py` (needs `pip install reportlab`) | `report/Report.pdf` (`--out` to change) | seconds |
| 6 Feature groups PDF | `python scripts/06_build_feature_groups_pdf.py` (needs `reportlab`) | `report/Feature_Groups.pdf` | seconds |

- The download needs no Hugging Face login (public dataset, CC BY-NC 4.0; don't redistribute the audio).
- Fixed seed 42 throughout. Each step writes its log to `logs/`.
- Check the other team's split CSV against ours: `.venv/bin/python scripts/02_make_folds.py --check their_split.csv`.
- Train on their split instead: `.venv/bin/python scripts/03_train_logreg.py --splits their_split.csv` (needs columns `id`, `fold`).

## Output files
| File | Contents |
|---|---|
| `clips.csv` | the 10,636 clips: id, speaker_id, style, text, sample_rate, duration_s |
| `features/egemaps.parquet` | id, speaker_id, style, duration_s + the 88 eGeMAPS features |
| `splits/folds.csv` | id, speaker_id, style, fold (1-4 = test fold, fold k holds out ex0k), random_split (train/test) |
| `models/logreg_fold{1-4}.joblib` | fitted scaler + logistic regression per fold (sklearn Pipeline, C = 1) |
| `models/logreg_random_split.joblib` | same, trained on the random split |
| `results/metrics_per_fold.csv` | accuracy, macro-F1, C per fold, mean, sd, random split |
| `results/per_style.csv` | precision, recall, F1, support per style per fold (+ random split) |
| `results/predictions.csv` | every test prediction: split, id, speaker_id, style, pred, p_<style> × 7 |
| `results/coefficients_fold{1-4}.csv` | 7 styles × 88 standardised coefficients per fold |
| `results/confusion_*.csv / .png` | confusion matrices per fold and summed over the 4 folds |
| `results/c_sensitivity_macro_f1.json` | inner-CV macro-F1 for every C, per fold (sensitivity check only) |
| `results/c_tuned_vs_fixed.csv` | test results with C tuned per fold vs fixed C = 1 |
| `results/feature_groups.csv` | each of the 88 features → one of 7 groups, and whether a narrator can change it |
| `results/group_contributions_loao.csv` | per style and group: share of contribution (%) and signed contribution, mean and sd over 4 folds |
| `results/group_contributions_per_fold.csv` | the same per fold |
| `results/group_contributions_random.csv` | the same for the random-split model |
| `results/summary.txt`, `logs/` | printed report and the log of each step |
| `report/Report.pdf` | the report |
| `report/Feature_Groups.pdf` | reference sheet: the 7 feature groups and the exact feature labels in each |

### Loading a model to explain or test it
```python
import joblib, pandas as pd
model = joblib.load("models/logreg_fold3.joblib")      # Pipeline(scaler -> logreg)
X = pd.read_parquet("features/egemaps.parquet")
feature_cols = X.columns[4:]                            # the 88 eGeMAPS columns
proba = model.predict_proba(X[feature_cols])            # columns follow model.classes_
scaler, logreg = model.named_steps["scaler"], model.named_steps["logreg"]
k = list(logreg.classes_).index("sad")
contrib = scaler.transform(X[feature_cols]) * logreg.coef_[k]   # exact per-feature contributions to "sad" (= SHAP)
```

## Choices
- **Data:** `ylacombe/expresso`, config `read`; keep 7 styles (confused, default, enunciated, happy, laughing, sad, whisper) = 10,636 clips. Dropped emphasis, essentials, longform (all "default substyles" per the official README) and singing; team decision. Longform ids collide with 4 default ids, another reason to drop them before joining on `id`.
- **Features:** 48 → 16 kHz (polyphase), mono, **full clip** (no crop or pad), **no loudness normalisation**, openSMILE eGeMAPSv02 functionals (88).
- **Evaluation:** leave-one-actor-out, 4 folds (fold k tests on ex0k); plus a random 80/20 split stratified by style, to measure speaker leakage.
- **Model:** StandardScaler + multinomial logistic regression (lbfgs, max_iter 5000), fit on training actors only. **C fixed at 1 in every fold** (so coefficients are comparable across folds); C ∈ {0.001 … 100} is only scanned as a sensitivity check by inner leave-one-actor-out over the 3 training actors. Seed 42.
- **Explanations:** contribution of feature f to style s = coef[s, f] × (x_f − training mean) / training sd (exact; equals SHAP for this model). Grouped into 7 feature families for actionable recourse (`scripts/feature_groups.py`).
- **Speaker sex:** official sources only say "2 males, 2 females" with no per-ID mapping, so results are reported per actor (proposal §7).

## Dataset credit and licence
This project uses the **Expresso** dataset (Meta AI), via the Hugging Face copy `ylacombe/expresso`, licensed
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (non-commercial use only).
The audio is **not** included in this repository; download it with `scripts/00_download_expresso.py`.
`clips.csv` contains the dataset's transcripts and metadata, and `features/`, `models/` and `results/` are
derived from it; they are shared here under the same non-commercial terms, for coursework only.

> Tu Anh Nguyen, Wei-Ning Hsu, Antony D'Avirro, Bowen Shi, Itai Gat, Maryam Fazel-Zarani, Tal Remez, Jade Copet,
> Gabriel Synnaeve, Michael Hassid, Felix Kreuk, Yossi Adi, Emmanuel Dupoux. 2023. *EXPRESSO: A Benchmark and
> Analysis of Discrete Expressive Speech Resynthesis.* Interspeech 2023. https://arxiv.org/abs/2308.05725
