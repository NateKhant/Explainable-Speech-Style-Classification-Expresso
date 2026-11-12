"""Explanation mechanism: per-style contribution of 7 feature groups (actionable recourse).

For a logistic regression on standardised features, the exact contribution of
feature f to the score (logit) of style s for clip x is
    coef[s, f] * (x_f - training mean_f) / training sd_f
which is also the SHAP value of this linear model (interventional, training data
as background). The training mean/sd come from each model's own scaler, so the
test actor never influences the explanation.

For every style s, over the TEST clips whose true style is s:
  share  = mean |contribution| summed within a group, as % of the sum over all groups
           (each style's row sums to 100%)
  signed = mean signed contribution of the group to the logit of s
           (+ pushes the clip toward s, - pushes it away)
Leave-one-actor-out: computed per fold, then averaged over the 4 folds (sd across
folds reported). Random split: the random-split model on its test clips.

Outputs (results/):
  feature_groups.csv               feature -> group, changeable
  group_contributions_loao.csv     style, group, share_mean, share_sd, signed_mean, signed_sd
  group_contributions_random.csv   style, group, share, signed

Usage:
    .venv/bin/python scripts/04_group_contributions.py
"""

import joblib
import numpy as np
import pandas as pd

from config import FEATURES, FOLDS_CSV, MODELS, RESULTS, STYLES
from feature_groups import CHANGEABLE, GROUP_NAMES, feature_groups

META = ["id", "speaker_id", "style", "duration_s"]


def group_contributions(model, X: np.ndarray, y: np.ndarray, groups: dict, feat_cols) -> pd.DataFrame:
    scaler, lr = model.named_steps["scaler"], model.named_steps["logreg"]
    z = scaler.transform(X)  # (x - training mean) / training sd
    gidx = {g: [i for i, f in enumerate(feat_cols) if groups[f] == g] for g in GROUP_NAMES}
    rows = []
    for s in STYLES:
        k = list(lr.classes_).index(s)
        contrib = z[y == s] * lr.coef_[k]  # (clips of style s, 88) contributions to logit of s
        absg = {g: np.abs(contrib[:, idx]).sum(axis=1).mean() for g, idx in gidx.items()}
        total = sum(absg.values())
        for g, idx in gidx.items():
            rows.append(dict(style=s, group=g, share=100 * absg[g] / total,
                             signed=contrib[:, idx].sum(axis=1).mean(), n_clips=int((y == s).sum())))
    return pd.DataFrame(rows)


def main() -> None:
    feats = pd.read_parquet(FEATURES)
    feat_cols = [c for c in feats.columns if c not in META]
    groups = feature_groups(feat_cols)
    fg = pd.DataFrame({"feature": feat_cols, "group": [groups[f] for f in feat_cols]})
    fg["changeable"] = fg["group"].map(CHANGEABLE)
    fg.to_csv(RESULTS / "feature_groups.csv", index=False)
    print("Features per group:", fg["group"].value_counts().reindex(GROUP_NAMES).to_dict(), "total", len(fg))

    splits = pd.read_csv(FOLDS_CSV)
    df = feats.merge(splits[["id", "fold", "random_split"]], on="id", validate="one_to_one")

    per_fold = []
    for k in sorted(df["fold"].unique()):
        te = df["fold"] == k
        model = joblib.load(MODELS / f"logreg_fold{k}.joblib")
        r = group_contributions(model, df.loc[te, feat_cols].to_numpy(), df.loc[te, "style"].to_numpy(),
                                groups, feat_cols)
        r.insert(0, "fold", k)
        per_fold.append(r)
    pf = pd.concat(per_fold)
    loao = (pf.groupby(["style", "group"])
              .agg(share_mean=("share", "mean"), share_sd=("share", "std"),
                   signed_mean=("signed", "mean"), signed_sd=("signed", "std"))
              .reset_index())
    pf.to_csv(RESULTS / "group_contributions_per_fold.csv", index=False)
    loao.to_csv(RESULTS / "group_contributions_loao.csv", index=False)

    te = df["random_split"] == "test"
    model = joblib.load(MODELS / "logreg_random_split.joblib")
    rnd = group_contributions(model, df.loc[te, feat_cols].to_numpy(), df.loc[te, "style"].to_numpy(),
                              groups, feat_cols)
    rnd.to_csv(RESULTS / "group_contributions_random.csv", index=False)

    for name, d, col in [("Leave-one-actor-out (mean of 4 folds)", loao, "share_mean"),
                         ("Random split", rnd, "share")]:
        print(f"\n{name}: share of |contribution| (%), rows = true style")
        print(d.pivot(index="style", columns="group", values=col)[GROUP_NAMES].round(1).to_string())
    print("\nLeave-one-actor-out: mean signed contribution to the style's logit")
    print(loao.pivot(index="style", columns="group", values="signed_mean")[GROUP_NAMES].round(2).to_string())


if __name__ == "__main__":
    main()
