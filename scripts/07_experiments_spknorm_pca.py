"""Experiments on the LR path: per-speaker normalisation x group-PCA inputs.

Grid (6 configurations, each on the 4 leave-one-actor-out folds + the random split):
  normalisation  raw      : features as extracted (same as 03_train_logreg.py)
                 spknorm  : x - mean of this speaker's calibration clips (mean-only)
  inputs         all88    : the 88 eGeMAPS features
                 pc1      : one PCA score per feature group (9 groups -> 9 inputs)
                 kpc90    : per group, the fewest PCs that explain >= 90% of that group's variance

Calibration: for every actor, 50 default clips (seed 42) form the "reference read" used to
compute that actor's mean. They are never used for training or testing, in ANY configuration,
so all 6 configurations are scored on exactly the same clips.

Groups: the 7 groups of feature_groups.py with Formants split into frequency / bandwidth /
amplitude (9 groups). Inside each fold, the per-group StandardScaler and PCA are fit on the
training actors only. Each PC1 is oriented so that "+" means more of a named anchor feature
(e.g. Loudness: + = louder); this changes no prediction, only how a coefficient reads.

Model after the inputs: StandardScaler + multinomial logistic regression, C = 1, seed 42
(identical to the baseline).

Extra measurements:
  - coefficient stability across folds (all88, pc1): mean pairwise Spearman correlation of the
    7 x p coefficient matrices of the 4 fold models (PC signs are anchored, so comparable)
  - PCA diagnostics per group and fold: variance explained by PC1, k for 90%, PC1 direction
    similarity across folds (|cosine| with the fold-1 direction)
  - multiclass Brier score of the predicted probabilities, per fold and on the random split
    (mean over clips of sum_k (p_k - 1[y = k])^2; lower is better, uniform guess = 0.857)
  - speaker-ID probe: how well an LR predicts WHICH actor spoke from the inputs
    (5-fold stratified CV); lower = less speaker identity left in the inputs

Outputs: results/experiments/ and splits/calibration.csv. Existing results/ files are not touched.

Usage (from the "Expresso Project" folder):
    .venv/bin/python scripts/07_experiments_spknorm_pca.py
"""

import importlib
import io
import itertools
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import C_FIXED, FEATURES, FOLDS_CSV, RESULTS, ROOT, SEED, STYLES
from feature_groups import CHANGEABLE, group_of

plot_confusion = importlib.import_module("03_train_logreg").plot_confusion

OUT = RESULTS / "experiments"
CALIB_CSV = ROOT / "splits" / "calibration.csv"
N_CALIB = 50          # default clips per actor used as the speaker's reference read
VAR_TARGET = 0.90     # kpc90: keep PCs until this share of the group's variance is explained
META = ["id", "speaker_id", "style", "duration_s"]

FORMANT_SPLIT = [("frequency", "Formant frequency"), ("bandwidth", "Formant bandwidth"),
                 ("amplitude", "Formant amplitude")]
GROUPS9 = ["Rate & pauses", "Loudness", "Pitch movement", "Pitch level", "Voice quality",
           "Spectral / timbre", "Formant frequency", "Formant bandwidth", "Formant amplitude"]
CHANGEABLE9 = {**CHANGEABLE, **{name: "No" for _, name in FORMANT_SPLIT}}
# "+" direction of each group's PC1 = more of this feature
ANCHOR = {
    "Rate & pauses": ("VoicedSegmentsPerSec", "more voiced segments per second (faster)"),
    "Loudness": ("loudness_sma3_amean", "louder"),
    "Pitch movement": ("F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2", "wider pitch range"),
    "Pitch level": ("F0semitoneFrom27.5Hz_sma3nz_amean", "higher pitch"),
    "Voice quality": ("HNRdBACF_sma3nz_amean", "higher harmonics-to-noise (less breathy)"),
    "Spectral / timbre": ("spectralFlux_sma3_amean", "more spectral flux"),
    "Formant frequency": ("F1frequency_sma3nz_amean", "higher F1"),
    "Formant bandwidth": ("F1bandwidth_sma3nz_amean", "wider F1 bandwidth"),
    "Formant amplitude": ("F1amplitudeLogRelF0_sma3nz_amean", "stronger F1 amplitude"),
}
NORMS = ["raw", "spknorm"]
INPUTS = ["all88", "pc1", "kpc90"]


def group9_of(feature: str) -> str:
    g = group_of(feature)
    if g == "Formants":
        return next(name for key, name in FORMANT_SPLIT if key in feature)
    return g


class GroupPCA(BaseEstimator, TransformerMixin):
    """Per group: standardise, PCA, keep PC1 (mode='pc1') or the PCs reaching VAR_TARGET."""

    def __init__(self, groups, mode="pc1"):
        self.groups = groups  # list of (group name, column indices, anchor column index)
        self.mode = mode

    def fit(self, X, y=None):
        self.parts_ = []
        for name, idx, anchor in self.groups:
            sc = StandardScaler().fit(X[:, idx])
            pca = PCA(random_state=SEED).fit(sc.transform(X[:, idx]))
            cum = np.cumsum(pca.explained_variance_ratio_)
            k = 1 if self.mode == "pc1" else int(np.searchsorted(cum, VAR_TARGET) + 1)
            comps = pca.components_[:k].copy()
            a = list(idx).index(anchor)
            if comps[0, a] < 0:          # PC1: "+" = more of the anchor feature
                comps[0] *= -1
            for j in range(1, k):        # other PCs: largest |loading| positive (not interpreted)
                if comps[j, np.argmax(np.abs(comps[j]))] < 0:
                    comps[j] *= -1
            self.parts_.append(dict(name=name, idx=idx, scaler=sc, mean=pca.mean_, comps=comps, k=k,
                                    evr1=pca.explained_variance_ratio_[0], cum_k=cum[k - 1],
                                    anchor_loading=comps[0, a]))
        return self

    def transform(self, X):
        return np.hstack([(p["scaler"].transform(X[:, p["idx"]]) - p["mean"]) @ p["comps"].T
                          for p in self.parts_])

    def names(self):
        return [p["name"] if p["k"] == 1 else f"{p['name']} PC{j + 1}"
                for p in self.parts_ for j in range(p["k"])]


def make_model(inputs, groups) -> Pipeline:
    steps = [] if inputs == "all88" else [("groups", GroupPCA(groups, "pc1" if inputs == "pc1" else "kpc90"))]
    steps += [("scaler", StandardScaler()),
              ("logreg", LogisticRegression(C=C_FIXED, max_iter=5000, random_state=SEED))]
    return Pipeline(steps)


def make_calibration(df: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for spk in sorted(df["speaker_id"].unique()):
        ids = sorted(df.loc[(df["speaker_id"] == spk) & (df["style"] == "default"), "id"])
        rows += [dict(id=i, speaker_id=spk) for i in rng.choice(ids, N_CALIB, replace=False)]
    return pd.DataFrame(rows).sort_values("id").reset_index(drop=True)


def brier(y_true, proba, classes) -> float:
    """Multiclass Brier score: mean over clips of sum_k (p_k - 1[y = k])^2. 0 = perfect, 2 = worst,
    6/7 = 0.857 for a uniform guess over the 7 styles."""
    onehot = (np.asarray(y_true)[:, None] == np.asarray(classes)[None, :]).astype(float)
    return float(np.mean(np.sum((proba - onehot) ** 2, axis=1)))


def mean_pairwise_spearman(mats) -> float:
    return float(np.mean([spearmanr(a.ravel(), b.ravel())[0] for a, b in itertools.combinations(mats, 2)]))


def main() -> None:
    feats = pd.read_parquet(FEATURES)
    feat_cols = [c for c in feats.columns if c not in META]
    assert len(feat_cols) == 88
    folds = pd.read_csv(FOLDS_CSV)
    df = feats.merge(folds[["id", "fold", "random_split"]], on="id", how="inner", validate="one_to_one")
    assert len(df) == len(feats) == len(folds)

    calib = make_calibration(df)
    CALIB_CSV.write_text(calib.to_csv(index=False))
    is_cal = df["id"].isin(calib["id"]).to_numpy()
    assert is_cal.sum() == N_CALIB * df["speaker_id"].nunique()

    # speaker means from the calibration clips only (mean-only normalisation)
    spk_mean = df.loc[is_cal].groupby("speaker_id")[feat_cols].mean()
    X_raw = df[feat_cols].to_numpy()
    X_norm = X_raw - spk_mean.loc[df["speaker_id"]].to_numpy()
    Xs = {"raw": X_raw, "spknorm": X_norm}

    gidx = {g: [i for i, f in enumerate(feat_cols) if group9_of(f) == g] for g in GROUPS9}
    assert sum(map(len, gidx.values())) == 88
    groups = [(g, gidx[g], feat_cols.index(ANCHOR[g][0])) for g in GROUPS9]
    pd.DataFrame([dict(feature=f, group=group9_of(f), changeable=CHANGEABLE9[group9_of(f)])
                  for f in feat_cols]).to_csv(OUT_mk() / "feature_groups9.csv", index=False)

    use = ~is_cal                          # every evaluation uses the non-calibration clips only
    d = df.loc[use].reset_index(drop=True)
    y, spk = d["style"].to_numpy(), d["speaker_id"].to_numpy()

    log = io.StringIO()
    metric_rows, style_rows, pca_rows, coef_rows, stab_rows, probe_rows = [], [], [], [], [], []
    pc1_dirs = {}  # (norm, group) -> list of PC1 loading vectors per fold

    for norm, inputs in itertools.product(NORMS, INPUTS):
        X = Xs[norm][use]
        cfg = f"{norm}|{inputs}"
        coef_mats = []
        cm_sum = np.zeros((7, 7), dtype=int)
        for k in sorted(d["fold"].unique()):
            te = (d["fold"] == k).to_numpy()
            model = make_model(inputs, groups).fit(X[~te], y[~te])
            p = model.predict(X[te])
            n_in = model.named_steps["scaler"].n_features_in_
            metric_rows.append(dict(norm=norm, inputs=inputs, split=f"fold{k}", test_speaker=spk[te][0],
                                    n_inputs=n_in, n_train=int((~te).sum()), n_test=int(te.sum()),
                                    accuracy=accuracy_score(y[te], p), macro_f1=f1_score(y[te], p, average="macro"),
                                    brier=brier(y[te], model.predict_proba(X[te]), model.classes_)))
            pr, rc, f1, _ = precision_recall_fscore_support(y[te], p, labels=STYLES, zero_division=0)
            style_rows += [dict(norm=norm, inputs=inputs, split=f"fold{k}", test_speaker=spk[te][0], style=s,
                                precision=a, recall=b, f1=c) for s, a, b, c in zip(STYLES, pr, rc, f1)]
            cm_sum += confusion_matrix(y[te], p, labels=STYLES)
            lr = model.named_steps["logreg"]
            assert list(lr.classes_) == STYLES
            coef_mats.append(lr.coef_.copy())
            if inputs != "all88":
                gp = model.named_steps["groups"]
                for part in gp.parts_:
                    pca_rows.append(dict(norm=norm, inputs=inputs, fold=k, group=part["name"],
                                         n_features=len(part["idx"]), k=part["k"], evr_pc1=part["evr1"],
                                         var_explained_k=part["cum_k"], anchor=ANCHOR[part["name"]][0],
                                         anchor_loading=part["anchor_loading"]))
                    if inputs == "pc1":
                        pc1_dirs.setdefault((norm, part["name"]), []).append(part["comps"][0])
                if inputs == "pc1":
                    coef_rows += [dict(norm=norm, fold=k, style=s, group=g, coef=lr.coef_[i, j])
                                  for i, s in enumerate(STYLES) for j, g in enumerate(gp.names())]

        pd.DataFrame(cm_sum, index=STYLES, columns=STYLES).to_csv(OUT / f"confusion_{norm}_{inputs}.csv")
        fm = pd.DataFrame([r for r in metric_rows if r["norm"] == norm and r["inputs"] == inputs])
        plot_confusion(cm_sum, f"{norm} / {inputs}: 4 held-out actors - acc {fm['accuracy'].mean():.1%}",
                       OUT / f"confusion_{norm}_{inputs}.png")
        if inputs in ("all88", "pc1"):
            stab_rows.append(dict(norm=norm, inputs=inputs, coef_spearman_mean_pairwise=mean_pairwise_spearman(coef_mats)))

        # random 80/20 split (speakers mixed), calibration clips excluded as everywhere else
        tr = (d["random_split"] == "train").to_numpy()
        model = make_model(inputs, groups).fit(X[tr], y[tr])
        p = model.predict(X[~tr])
        metric_rows.append(dict(norm=norm, inputs=inputs, split="random_80_20", test_speaker="all",
                                n_inputs=model.named_steps["scaler"].n_features_in_, n_train=int(tr.sum()),
                                n_test=int((~tr).sum()), accuracy=accuracy_score(y[~tr], p),
                                macro_f1=f1_score(y[~tr], p, average="macro"),
                                brier=brier(y[~tr], model.predict_proba(X[~tr]), model.classes_)))

        # speaker-ID probe: predict the actor from the same inputs
        cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
        acc = cross_val_score(make_model(inputs, groups), X, spk, cv=cv, scoring="accuracy", n_jobs=-1)
        probe_rows.append(dict(norm=norm, inputs=inputs, speaker_id_accuracy=acc.mean(), sd=acc.std(ddof=1)))

    m = pd.DataFrame(metric_rows)
    summ = (m[m["split"].str.startswith("fold")].groupby(["norm", "inputs"], sort=False)
            .agg(n_inputs_mean=("n_inputs", "mean"), acc_mean=("accuracy", "mean"), acc_sd=("accuracy", "std"),
                 f1_mean=("macro_f1", "mean"), f1_sd=("macro_f1", "std"),
                 brier_mean=("brier", "mean"), brier_sd=("brier", "std")).reset_index())
    rnd = m[m["split"] == "random_80_20"].set_index(["norm", "inputs"])
    summ["random_acc"] = [rnd.loc[(a, b), "accuracy"] for a, b in zip(summ["norm"], summ["inputs"])]
    summ["random_brier"] = [rnd.loc[(a, b), "brier"] for a, b in zip(summ["norm"], summ["inputs"])]
    summ["gap_random_minus_loao"] = summ["random_acc"] - summ["acc_mean"]

    for (norm, g), vecs in pc1_dirs.items():
        sims = [abs(float(np.dot(vecs[0], v))) for v in vecs[1:]]
        for r in pca_rows:
            if r["norm"] == norm and r["group"] == g and r["inputs"] == "pc1":
                r["pc1_abs_cos_vs_fold1_min"] = min(sims)

    m.to_csv(OUT / "metrics.csv", index=False)
    summ.to_csv(OUT / "summary_by_config.csv", index=False)
    pd.DataFrame(style_rows).to_csv(OUT / "per_style.csv", index=False)
    pd.DataFrame(pca_rows).to_csv(OUT / "pca_groups.csv", index=False)
    pd.DataFrame(coef_rows).to_csv(OUT / "coefficients_pc1.csv", index=False)
    pd.DataFrame(stab_rows).to_csv(OUT / "coef_stability.csv", index=False)
    pd.DataFrame(probe_rows).to_csv(OUT / "speaker_probe.csv", index=False)

    with redirect_stdout(log):
        print(f"Clips: {len(df)} total, {is_cal.sum()} calibration (excluded), {len(d)} evaluated; seed {SEED}, C = {C_FIXED}\n")
        print("== Leave-one-actor-out, mean over 4 folds ==")
        print(summ.round(3).to_string(index=False))
        print("\n== Accuracy per held-out actor ==")
        print(m[m["split"].str.startswith("fold")].pivot_table(index=["norm", "inputs"], columns="test_speaker",
              values="accuracy", sort=False).round(3).to_string())
        print("\n== Recall per style, mean over folds ==")
        ps = pd.DataFrame(style_rows)
        print(ps.pivot_table(index="style", columns=["norm", "inputs"], values="recall", sort=False).round(3).to_string())
        print("\n== Coefficient stability across folds (mean pairwise Spearman) ==")
        print(pd.DataFrame(stab_rows).round(3).to_string(index=False))
        print("\n== Speaker-ID probe (5-fold CV accuracy; chance 0.25) ==")
        print(pd.DataFrame(probe_rows).round(3).to_string(index=False))
        print("\n== PCA per group (mean over folds) ==")
        pg = pd.DataFrame(pca_rows)
        print(pg.groupby(["norm", "inputs", "group"], sort=False)[["n_features", "k", "evr_pc1", "var_explained_k"]]
              .mean().round(3).to_string())
    (OUT / "summary.txt").write_text(log.getvalue())
    print(log.getvalue())


def OUT_mk():
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT


if __name__ == "__main__":
    main()
