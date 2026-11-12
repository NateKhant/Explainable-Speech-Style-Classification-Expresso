"""Step 4 (baseline path): logistic regression on the 88 eGeMAPS features.

Main evaluation: leave-one-actor-out, 4 folds (train on 3 actors, test on the 4th).
Inside each fold:
  - StandardScaler fit on the 3 training actors only
  - multinomial logistic regression (lbfgs), fixed C = C_FIXED (1.0) in every fold
    so coefficients are comparable across folds; no class weights (classes balanced)
  - sensitivity check only: macro-F1 for every C in C_GRID by an inner
    leave-one-actor-out over the 3 training actors (never the test actor)
Beside it: a random 80/20 split stratified by style, same fixed C (sensitivity
scan by 5-fold CV on the training part). A big gap between the two suggests the
model leans on the speaker.

Outputs (results/):
  metrics_per_fold.csv     accuracy, macro-F1, C per fold (+ mean/std row) and random split
  c_sensitivity_macro_f1.json  inner-CV macro-F1 for every C in C_GRID (not used to pick C)
  per_style.csv            precision / recall / F1 per style per fold
  predictions.csv          test predictions + class probabilities per clip
  coefficients_fold{k}.csv standardised coefficients (7 styles x 88 features)
  confusion_*.csv / .png   per fold and summed over the 4 folds (row = true style)
  summary.txt              the printed report
models/logreg_fold{k}.joblib  fitted scaler + model pipeline per fold

Usage:
    .venv/bin/python scripts/03_train_logreg.py [--splits splits/folds.csv]
    (--splits accepts the other team's CSV if it has columns id + fold)
"""

import argparse
from pathlib import Path
import io
import json
from contextlib import redirect_stdout

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import C_FIXED, C_GRID, FEATURES, FOLDS_CSV, MODELS, RESULTS, ROOT, SEED, STYLES

META = ["id", "speaker_id", "style", "duration_s"]

# Chart tokens (dataviz reference palette, light theme): sequential blue ramp.
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def make_pipeline(c: float = 1.0) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(C=c, max_iter=5000, random_state=SEED)),
    ])


def c_sensitivity(X, y, cv, groups=None) -> dict:
    """Mean inner-CV macro-F1 for every C in C_GRID. Reported only; does not choose C."""
    search = GridSearchCV(make_pipeline(), {"logreg__C": C_GRID}, scoring="f1_macro",
                          cv=cv, n_jobs=-1, refit=False)
    search.fit(X, y, groups=groups)
    return dict(zip(map(str, C_GRID), np.round(search.cv_results_["mean_test_score"], 4).tolist()))


def fit_fixed(X, y) -> Pipeline:
    return make_pipeline(C_FIXED).fit(X, y)


def tuned_c_row(split: str, scores: dict, Xtr, ytr, Xte, yte) -> dict:
    """Comparison only: refit with the C that scored best in the inner scan (the
    per-fold tuning used in the first report version) and score it on the test set."""
    best = max(C_GRID, key=lambda c: scores[str(c)])
    p = make_pipeline(best).fit(Xtr, ytr).predict(Xte)
    return dict(split=split, tuned_C=best, accuracy=accuracy_score(yte, p),
                macro_f1=f1_score(yte, p, average="macro"))


def plot_confusion(cm: np.ndarray, title: str, path) -> None:
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
    rec = cm / cm.sum(axis=1, keepdims=True)  # row-normalised = recall per true style
    fig, ax = plt.subplots(figsize=(6.4, 5.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    im = ax.imshow(rec, cmap=cmap, vmin=0, vmax=1)
    n = len(STYLES)
    ax.set_xticks(range(n), STYLES, rotation=35, ha="right", color=INK_2, fontsize=9)
    ax.set_yticks(range(n), STYLES, color=INK_2, fontsize=9)
    ax.set_xlabel("Predicted style", color=INK_2)
    ax.set_ylabel("True style", color=INK_2)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    ax.set_xticks(np.arange(-.5, n + 1), minor=True)
    ax.set_yticks(np.arange(-.5, n + 1), minor=True)
    ax.set_xlim(-.5, n - .5)
    ax.set_ylim(n - .5, -.5)
    ax.grid(which="minor", color=SURFACE, linewidth=2)  # 2px surface gap between cells
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(n):
        for j in range(n):
            v = rec[i, j]
            ax.text(j, i, f"{v:.0%}\n{cm[i, j]}", ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if v > 0.55 else (INK if v > 0.005 else MUTED))
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, format=lambda x, _: f"{x:.0%}")
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUTED, labelsize=8, length=0)
    cb.set_label("Share of true-style clips (recall)", color=MUTED, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=str(FOLDS_CSV))
    args = ap.parse_args()

    feats = pd.read_parquet(FEATURES)
    feat_cols = [c for c in feats.columns if c not in META]
    assert len(feat_cols) == 88, len(feat_cols)

    splits = pd.read_csv(args.splits)
    keep_cols = ["id", "fold"] + (["random_split"] if "random_split" in splits else [])
    df = feats.merge(splits[keep_cols], on="id", how="inner", validate="one_to_one")
    assert len(df) == len(feats) == len(splits), (len(df), len(feats), len(splits))
    for k, g in df.groupby("fold"):
        assert g["speaker_id"].nunique() == 1, f"fold {k} test set has several speakers"

    X, y = df[feat_cols].to_numpy(), df["style"].to_numpy()
    RESULTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    log = io.StringIO()
    fold_rows, style_rows, preds, tuning = [], [], [], {}
    cm_sum = np.zeros((len(STYLES), len(STYLES)), dtype=int)
    cmp_rows = []  # comparison: C picked per fold by the sensitivity scan (first version of the report)

    with redirect_stdout(log):
        print(f"Features: {feats.shape[0]} clips x {len(feat_cols)} eGeMAPS features")
        sp = Path(args.splits).resolve()
        shown = sp.relative_to(ROOT) if sp.is_relative_to(ROOT) else sp.name  # no local paths in outputs
        print(f"Splits: {shown}\nC = {C_FIXED} (fixed); sensitivity grid {C_GRID}; seed {SEED}\n")
        print("== Leave-one-actor-out ==")

    for k in sorted(df["fold"].unique()):
        te = (df["fold"] == k).to_numpy()
        tr = ~te
        test_spk = df.loc[te, "speaker_id"].iloc[0]
        tuning[f"fold{k}"] = c_sensitivity(X[tr], y[tr], LeaveOneGroupOut(), groups=df.loc[tr, "speaker_id"])
        model = fit_fixed(X[tr], y[tr])
        cmp_rows.append(tuned_c_row(f"fold{k}", tuning[f"fold{k}"], X[tr], y[tr], X[te], y[te]))

        p = model.predict(X[te])
        proba = model.predict_proba(X[te])
        acc, mf1 = accuracy_score(y[te], p), f1_score(y[te], p, average="macro")
        fold_rows.append(dict(split=f"fold{k}", test_speaker=test_spk, n_train=int(tr.sum()),
                              n_test=int(te.sum()), C=C_FIXED, accuracy=acc, macro_f1=mf1))

        pr, rc, f1, sup = precision_recall_fscore_support(y[te], p, labels=STYLES, zero_division=0)
        style_rows += [dict(split=f"fold{k}", test_speaker=test_spk, style=s, precision=a, recall=b,
                            f1=c, support=int(d)) for s, a, b, c, d in zip(STYLES, pr, rc, f1, sup)]

        cm = confusion_matrix(y[te], p, labels=STYLES)
        cm_sum += cm
        pd.DataFrame(cm, index=STYLES, columns=STYLES).to_csv(RESULTS / f"confusion_fold{k}.csv")
        plot_confusion(cm, f"Fold {k}: test actor {test_spk} - acc {acc:.1%}, macro-F1 {mf1:.3f}",
                       RESULTS / f"confusion_fold{k}.png")

        classes = list(model.classes_)
        pf = df.loc[te, ["id", "speaker_id", "style"]].copy()
        pf.insert(0, "split", f"fold{k}")
        pf["pred"] = p
        for j, s in enumerate(classes):
            pf[f"p_{s}"] = proba[:, j]
        preds.append(pf)

        lr = model.named_steps["logreg"]
        pd.DataFrame(lr.coef_, index=classes, columns=feat_cols).to_csv(RESULTS / f"coefficients_fold{k}.csv")
        joblib.dump(model, MODELS / f"logreg_fold{k}.joblib")
        with redirect_stdout(log):
            print(f"fold {k} (test {test_spk}, n={te.sum()}): C={C_FIXED:g}  acc={acc:.3f}  macro-F1={mf1:.3f}")

    folds = pd.DataFrame(fold_rows)
    mean = folds[["accuracy", "macro_f1"]].mean()
    std = folds[["accuracy", "macro_f1"]].std(ddof=1)
    rows = fold_rows + [dict(split="mean_4folds", accuracy=mean["accuracy"], macro_f1=mean["macro_f1"]),
                        dict(split="std_4folds", accuracy=std["accuracy"], macro_f1=std["macro_f1"])]

    pd.DataFrame(cm_sum, index=STYLES, columns=STYLES).to_csv(RESULTS / "confusion_all_folds.csv")
    plot_confusion(cm_sum, f"All 4 held-out actors - acc {mean['accuracy']:.1%} (mean of folds)",
                   RESULTS / "confusion_all_folds.png")

    styles = pd.DataFrame(style_rows)
    with redirect_stdout(log):
        print(f"\nmean ± sd over 4 folds: acc {mean['accuracy']:.3f} ± {std['accuracy']:.3f}, "
              f"macro-F1 {mean['macro_f1']:.3f} ± {std['macro_f1']:.3f}")
        print("\nRecall per style (rows) per held-out actor (columns):")
        print(styles.pivot(index="style", columns="test_speaker", values="recall").round(3).to_string())
        print("\nF1 per style, mean over folds:")
        print(styles.groupby("style")["f1"].mean().round(3).to_string())

    if "random_split" in df:
        tr = (df["random_split"] == "train").to_numpy()
        te = ~tr
        tuning["random_split"] = c_sensitivity(X[tr], y[tr], StratifiedKFold(5, shuffle=True, random_state=SEED))
        model = fit_fixed(X[tr], y[tr])
        cmp_rows.append(tuned_c_row("random_80_20", tuning["random_split"], X[tr], y[tr], X[te], y[te]))
        p = model.predict(X[te])
        acc, mf1 = accuracy_score(y[te], p), f1_score(y[te], p, average="macro")
        rows.append(dict(split="random_80_20", test_speaker="all", n_train=int(tr.sum()), n_test=int(te.sum()),
                         C=C_FIXED, accuracy=acc, macro_f1=mf1))
        pr, rc, f1, sup = precision_recall_fscore_support(y[te], p, labels=STYLES, zero_division=0)
        styles = pd.concat([styles, pd.DataFrame([dict(split="random_80_20", test_speaker="all", style=s,
                            precision=a, recall=b, f1=c, support=int(d)) for s, a, b, c, d in zip(STYLES, pr, rc, f1, sup)])])
        joblib.dump(model, MODELS / "logreg_random_split.joblib")
        with redirect_stdout(log):
            print(f"\n== Random 80/20 split (speakers mixed) ==\nC={C_FIXED:g} acc={acc:.3f} macro-F1={mf1:.3f}")
            print(f"Gap random - leave-one-actor-out: acc {acc - mean['accuracy']:+.3f}, "
                  f"macro-F1 {mf1 - mean['macro_f1']:+.3f}")

    pd.DataFrame(rows).to_csv(RESULTS / "metrics_per_fold.csv", index=False)
    styles.to_csv(RESULTS / "per_style.csv", index=False)
    pd.concat(preds).to_csv(RESULTS / "predictions.csv", index=False)
    (RESULTS / "c_sensitivity_macro_f1.json").write_text(json.dumps(tuning, indent=2))
    cmp = pd.DataFrame(cmp_rows)
    fixed = pd.DataFrame(rows).set_index("split")
    cmp["fixed_C"] = C_FIXED
    cmp["accuracy_fixed_C"] = cmp["split"].map(fixed["accuracy"])
    cmp["macro_f1_fixed_C"] = cmp["split"].map(fixed["macro_f1"])
    cmp.to_csv(RESULTS / "c_tuned_vs_fixed.csv", index=False)
    with redirect_stdout(log):
        print("\n== Comparison: C tuned per fold vs fixed C ==")
        print(cmp.round(4).to_string(index=False))
    (RESULTS / "summary.txt").write_text(log.getvalue())
    print(log.getvalue())


if __name__ == "__main__":
    main()
