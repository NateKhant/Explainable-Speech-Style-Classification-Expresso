"""Build report/Experiments_Report.pdf from results/experiments/.

Run after 07_experiments_spknorm_pca.py. Needs reportlab (pip install reportlab).
Every number in the text is read from the result files, so the PDF regenerates from code.
Usage (from the "Expresso Project" folder):
    python scripts/08_build_experiments_report.py [--out report/Experiments_Report.pdf]
"""

import argparse
import importlib
import sys
from pathlib import Path

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import KeepTogether, SimpleDocTemplate, Spacer, Table

sys.path.insert(0, str(Path(__file__).resolve().parent))
R = importlib.import_module("05_build_report")          # reuse styles and table helpers
P, bullets, table, img, S, MUTED = R.P, R.bullets, R.table, R.img, R.S, R.MUTED
exp = importlib.import_module("07_experiments_spknorm_pca")

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "experiments"
OUT = ROOT / "report" / "Experiments_Report.pdf"
STYLES = ["confused", "default", "enunciated", "happy", "laughing", "sad", "whisper"]
SPEAKERS = ["ex01", "ex02", "ex03", "ex04"]
NORM_LABEL = {"raw": "Raw", "spknorm": "Speaker-norm"}
UNIFORM_BRIER = 6 / 7   # (1 - 1/7)^2 + 6 (1/7)^2: Brier score of predicting 1/7 for every style
INPUT_LABEL = {"all88": "88 features", "pc1": "PC1 per group", "kpc90": "k PCs (90%)"}


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(2 * cm, 1.2 * cm, "Black Box group - LR path experiments: speaker normalisation and group PCA")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


def pct(x):
    return f"{x * 100:.1f}%"


def pts(x):
    return f"{x * 100:+.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    out = ap.parse_args().out

    m = pd.read_csv(RES / "metrics.csv")
    sm = pd.read_csv(RES / "summary_by_config.csv").set_index(["norm", "inputs"])
    ps = pd.read_csv(RES / "per_style.csv")
    pg = pd.read_csv(RES / "pca_groups.csv")
    co = pd.read_csv(RES / "coefficients_pc1.csv")
    st = pd.read_csv(RES / "coef_stability.csv").set_index(["norm", "inputs"])
    pr = pd.read_csv(RES / "speaker_probe.csv").set_index(["norm", "inputs"])
    fm = m[m["split"].str.startswith("fold")]
    acc_spk = fm.pivot_table(index=["norm", "inputs"], columns="test_speaker", values="accuracy")
    rec = ps.pivot_table(index="style", columns=["norm", "inputs"], values="recall")
    n_eval = int(fm.groupby(["norm", "inputs"])["n_test"].sum().iloc[0])

    base, spk = sm.loc[("raw", "all88")], sm.loc[("spknorm", "all88")]
    spc1, skpc, rpc1 = sm.loc[("spknorm", "pc1")], sm.loc[("spknorm", "kpc90")], sm.loc[("raw", "pc1")]
    b_spk0 = fm.pivot_table(index=["norm", "inputs"], columns="test_speaker", values="brier")
    d_ex03 = acc_spk.loc[("spknorm", "all88"), "ex03"] - acc_spk.loc[("raw", "all88"), "ex03"]
    others_max = max(abs(acc_spk.loc[("spknorm", "all88"), s] - acc_spk.loc[("raw", "all88"), s])
                     for s in SPEAKERS if s != "ex03")
    d_def = rec.loc["default", ("spknorm", "all88")] - rec.loc["default", ("raw", "all88")]

    W = A4[0] - 4 * cm
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=2 * cm,
                            title="LR path experiments: speaker normalisation and group PCA",
                            author="Aung Khant Maung")
    E = []
    E += [P("Speaker normalisation and group-level inputs for the eGeMAPS baseline", "title"),
          P("Black Box group (Explainable Data Science, AIT) - experiments on the logistic-regression path. "
            "Companion to Final_Report.pdf; the setup described there is not repeated.", "sub"),
          Spacer(1, 8)]

    E += [P("Summary", "h1")]
    E += bullets([
        f"<b>Speaker normalisation</b> (subtract each actor's mean over 50 reference <i>default</i> clips) raises "
        f"leave-one-actor-out accuracy from {pct(base.acc_mean)} to <b>{pct(spk.acc_mean)}</b> and macro-F1 from "
        f"{base.f1_mean:.3f} to <b>{spk.f1_mean:.3f}</b>, and <b>halves the spread across actors</b> "
        f"(sd {base.acc_sd * 100:.1f} to {spk.acc_sd * 100:.1f} points). The hardest actor, ex03, gains "
        f"{pts(d_ex03)} points; the other three move by at most {others_max * 100:.1f} points.",
        f"<b>One score per feature group</b> (PCA, 9 groups with formants split) costs accuracy: "
        f"{pct(spc1.acc_mean)} with 9 inputs against {pct(spk.acc_mean)} with 88 (speaker-normalised). Keeping "
        f"enough PCs for 90% of each group's variance (about {skpc.n_inputs_mean:.0f} inputs) recovers most of it "
        f"({pct(skpc.acc_mean)}).",
        f"<b>Group scores are more stable</b>: the coefficients of the 4 fold models agree with Spearman "
        f"{st.loc[('spknorm', 'pc1'), 'coef_spearman_mean_pairwise']:.2f} (9 group scores) against "
        f"{st.loc[('spknorm', 'all88'), 'coef_spearman_mean_pairwise']:.2f} (88 features).",
        f"<b>Speaker identity is reduced, not removed</b>: an LR that predicts the actor from the 88 features drops "
        f"from {pct(pr.loc[('raw', 'all88'), 'speaker_id_accuracy'])} to "
        f"{pct(pr.loc[('spknorm', 'all88'), 'speaker_id_accuracy'])} after normalisation (chance 25%).",
        f"<b>Probabilities improve too</b>: the Brier score falls from {base.brier_mean:.3f} to {spk.brier_mean:.3f} "
        f"(uniform guess {UNIFORM_BRIER:.3f}). With raw features the ex03 model is confidently wrong, at uniform-guess "
        f"level ({b_spk0.loc[('raw', 'all88'), 'ex03']:.3f}).",
        f"<b>Caveat</b>: default recall rises by {pts(d_def)} points partly by design, because the reference read "
        f"<i>is</i> default speech; default also becomes the most over-predicted label (Sections 4.1-4.2).",
    ])

    # ---------- 1. Background and design choices (the reasoning behind the experiments) ----------
    E += [P("1. Background and design choices", "h1"),
          P("This section records why the experiments were designed as they are: the problem they address, the "
            "options that were considered, and what was chosen. Sections 2-7 then report what was run and found.")]

    E += [P("1.1 The problem: every actor has a personal baseline", "h2"),
          P(f"The baseline scores {pct(base.acc_mean)} when an actor is held out but {pct(base.random_acc)} when every "
            "actor is seen in training, and the hardest held-out actor (ex03) is far below the others. Both point to the "
            "same cause: each actor has a personal baseline (their usual pitch, loudness and timbre), and the model "
            "sees <i>absolute</i> feature values. A value that is 'high' for one voice is 'normal' for another. The "
            "table shows this with the real median pitch per actor and style.")]
    feats = pd.read_parquet(exp.FEATURES)
    f0 = "F0semitoneFrom27.5Hz_sma3nz_percentile50.0"
    med = feats.groupby(["speaker_id", "style"])[f0].median().unstack()
    rows = [["Actor", "Default", "Happy", "Sad", "Happy minus<br/>own default", "Sad minus<br/>own default"]]
    for s in SPEAKERS:
        r = med.loc[s]
        rows.append([s, f"{r['default']:.1f}", f"{r['happy']:.1f}", f"{r['sad']:.1f}",
                     f"{r['happy'] - r['default']:+.1f}", f"{r['sad'] - r['default']:+.1f}"])
    E += [table(rows, [2.0 * cm] + [(W - 2.0 * cm) / 5] * 5, shade_rows=(4, 5)),
          P(f"Median of the clip-level median pitch (eGeMAPS {f0.replace('_', ' ')}, in semitones), all clips. Shaded: "
            "the same values relative to the actor's own default. Whisper is left out because it has almost no voiced "
            "pitch.", "cap")]
    hi_sad = med["sad"].idxmax()
    lo_happy = med["happy"].idxmin()
    happy_up = (med["happy"] - med["default"])
    E += bullets([
        f"In absolute terms the styles overlap across actors: {hi_sad}'s <i>sad</i> ({med.loc[hi_sad, 'sad']:.1f}) is "
        f"higher than {lo_happy}'s <i>happy</i> ({med.loc[lo_happy, 'happy']:.1f}). A model trained on absolute pitch "
        "learns thresholds that fit the training voices and misfire on a new one.",
        f"Relative to each actor's own default, the pattern is consistent: happy is higher for every actor "
        f"(+{happy_up.min():.1f} to +{happy_up.max():.1f} semitones), and sad is lower for "
        f"{int(((med['sad'] - med['default']) < 0).sum())} of the 4 "
        f"(the exception: {(med['sad'] - med['default']).idxmax()}).",
        "So the style lives mostly in <i>how far a take moves from the speaker's normal voice</i>, not in the "
        "absolute value. The existing StandardScaler does not fix this: it uses one mean and spread pooled over the "
        "training actors, which removes units, not voices.",
    ])

    E += [P("1.2 Per-speaker normalisation", "h2"),
          P("Each feature is rewritten relative to the speaker's own baseline: x' = x - mean<sub>speaker</sub> (and "
            "optionally divided by the speaker's spread). After this, 'happy' reads as '+n above my normal pitch' "
            "for every voice, and the model can learn one rule that transfers. The question is which clips define "
            "'this speaker's normal' for an actor the model has never heard. Three options were considered:")]
    rows = [["Option", "Reference clips", "Fit with the director scenario", "Risk", "Decision"],
            ["A: all clips", "All of the speaker's clips (labels not used)",
             "Weak: needs the whole session in advance",
             "Mild leak: every actor's styles are perfectly balanced, so the mean is 'the average of the 7 styles'",
             "Not run"],
            ["B: reference read", f"{exp.N_CALIB} of the speaker's default clips, held out",
             "Strong: the narrator reads a few neutral lines before the session; each take is judged against them",
             "Uses labelled default clips; they must never be tested on", "<b>Chosen</b>"],
            ["C: first k takes", "k random unlabelled clips (k = 10-100)",
             "Realistic: the first takes of a session", "Noisy for small k", "Not run"]]
    E += [table(rows, [2.3 * cm, 3.4 * cm, 4.2 * cm, 4.6 * cm, W - 14.5 * cm])]
    E += bullets([
        "<b>The trap in option B, and how it is avoided.</b> If a clip were both in the reference set and in the test "
        "set, it would sit exactly at zero after normalisation, and default accuracy would be inflated by "
        f"construction. So {exp.N_CALIB} default clips per actor are set aside as calibration only, for the training "
        "actors too (so training and test data are processed in the same way), and they are removed from every "
        "configuration, including the raw baseline, so that all results are compared on the same clips.",
        "<b>Mean-only, not mean and spread.</b> Dividing by the speaker's spread also equalises how expressive each "
        f"voice is, but a spread estimated from {exp.N_CALIB} clips is noisier than a mean, and the spread of a neutral "
        "read need not match the spread of expressive takes, so dividing by it could exaggerate the other styles. "
        "Mean-only was chosen; dividing by the spread was not tested.",
        "<b>What it changes for explanations.</b> Contributions stay exact (coefficient x value), but their meaning "
        "changes from 'the pitch is high' to 'the pitch is higher than this narrator's neutral read'. That is the "
        "director's language, and it weakens pitch level as a speaker-sex proxy.",
    ])

    E += [P("1.3 One score per feature group", "h2"),
          P("Final_Report.pdf already summarises the explanations per group: the model uses all 88 features and their "
            "contributions are added up per group afterwards. The question here was different: can each group be "
            "collapsed into <i>one number per clip</i> (a 'loudness score', a 'pitch movement score'), and can the model "
            "be trained on those numbers only, so that it thinks in groups? Four ways to build a group score were "
            "considered:")]
    rows = [["Method", "How it works", "For", "Against"],
            ["Average of z-scores", "Standardise, flip signs so features point the same way, average",
             "Simplest to explain", "Sign choices are manual; near-duplicate features get extra weight"],
            ["<b>PCA per group (chosen)</b>", "The direction that captures most of the group's variation",
             "Data-driven weights; reports how much of the group one score keeps",
             "Sign is arbitrary (fixed by an anchor); may not be clean for mixed groups"],
            ["Supervised sub-models", "A small LR per group, then an LR on their outputs",
             "Usually the most accurate", "7 numbers per group, not one scale; needs cross-fitting; harder to explain"],
            ["One representative feature", "e.g. mean loudness for Loudness", "Perfectly readable",
             "Throws away most of each group"]]
    E += [table(rows, [3.4 * cm, 4.6 * cm, 4.2 * cm, W - 12.2 * cm])]
    E += bullets([
        "<b>Rules for PCA.</b> It is fit on the training actors only (the same rule as the scaler), and each PC1 is "
        "oriented by a named anchor feature so that '+' always means the same thing (for example, Loudness: + = louder). "
        "Orientation does not change any prediction; it makes coefficients readable and comparable across folds.",
        "<b>Formants split into three.</b> The 18 formant features mix three different things: frequency (vowel and "
        "articulation), bandwidth and amplitude. One score for all three would be hard to name, so they became three "
        "groups of 6, giving 9 groups.",
        "<b>A middle option.</b> Besides one score per group (9 inputs), a version keeps, per group, the fewest "
        "components that explain 90% of its variance. This shows how much accuracy is lost through compression rather "
        "than through grouping itself.",
        "<b>Why it could be worth losing some accuracy.</b> (1) Stability: eGeMAPS features are highly correlated, so "
        "weight moves between near-duplicates from fold to fold; a few nearly independent scores should give steadier "
        "coefficients. (2) Fair comparison: a 33-feature group can collect a larger share of the contribution "
        "partly by counting; with one score each, groups compete on equal terms. (3) Actionability and the human "
        "study: a counterfactual over two dials (loudness, rate) is easier to act on than one over 17 features.",
        "<b>Caveat.</b> A group score is still abstract. 'Raise the loudness score by 0.8' must still be mapped to an "
        "audio edit (e.g. +3 dB gain) and checked by re-extracting the features; there are just fewer dials.",
    ])

    E += [P("1.4 The Brier score", "h2"),
          P("Accuracy and macro-F1 look only at the top prediction. The Brier score looks at the whole probability "
            "vector: for each clip it adds up, over the 7 styles, the squared difference between the predicted "
            "probability and what happened (1 for the true style, 0 for the others), then averages over clips. 0 is "
            f"perfect, 2 is fully confident and wrong, and predicting 1/7 for every style scores {UNIFORM_BRIER:.3f}. "
            "Lower is better. Worked example for one clip whose true style is sad:")]

    def brier_one(p_true, p_wrong_top=None):
        """Brier for one clip: p_true on the true style; the rest spread evenly over the other 6,
        or p_wrong_top on one wrong style and the remainder spread over the other 5."""
        if p_wrong_top is None:
            rest = [(1 - p_true) / 6] * 6
        else:
            rest = [p_wrong_top] + [(1 - p_true - p_wrong_top) / 5] * 5
        return (1 - p_true) ** 2 + sum(r ** 2 for r in rest)

    rows = [["Model output", "p(sad)", "Correct?", "Brier for this clip"],
            ["Confident and right", "0.90", "yes", f"{brier_one(0.90):.3f}"],
            ["Unsure but right", "0.40", "yes", f"{brier_one(0.40):.3f}"],
            ["Uniform guess", "0.14", "-", f"{brier_one(1 / 7):.3f}"],
            ["Confident and wrong (0.90 on happy)", "0.02", "no", f"{brier_one(0.02, 0.90):.3f}"]]
    E += [table(rows, [6.0 * cm, 2.2 * cm, 2.2 * cm, W - 10.4 * cm]),
          P("The remaining probability is spread evenly over the other styles. Accuracy counts the first two rows "
            "the same; the Brier score separates them and penalises the last row heavily.", "cap")]
    E += bullets([
        "<b>Why it matters here.</b> The director would act on a rule such as 'flag the take when the requested style "
        "gets less than 50%', and that only works if the probabilities can be trusted on a new voice. The deletion "
        "curves and counterfactuals planned for the explanations also work on predicted probabilities.",
        "Compared with log loss, the Brier score is bounded, so a few confidently wrong clips cannot dominate it.",
    ])

    E += [P("1.5 Two extra checks", "h2")]
    E += bullets([
        "<b>Speaker-ID probe.</b> A separate logistic regression is trained to name the actor (4 classes, chance 25%) "
        "from the same inputs. If normalisation removes speaker identity, this accuracy should fall. It measures the "
        "claim directly, and it bears on the ethics criterion because pitch level is a speaker-sex proxy.",
        "<b>Coefficient stability.</b> The 4 fold models are trained on different actors. If their coefficient matrices "
        "rank the inputs similarly (Spearman correlation between every pair of folds), the model's reasoning does not "
        "depend much on which actors it saw. This is the model-level side of the stability criterion.",
    ])

    E += [P("2. What was run", "h1")]
    E += bullets([
        f"<b>Reference read.</b> For each actor, {exp.N_CALIB} default clips (seed {exp.SEED}) are set aside as a "
        f"calibration set: the neutral read a narrator could record before a session. They are excluded from "
        f"training and testing in <i>every</i> configuration, so all configurations below are scored on the same "
        f"{n_eval:,} clips (the 88-feature raw row is therefore the baseline on these clips).",
        "<b>Speaker normalisation (mean-only).</b> Every feature is replaced by its difference from the speaker's "
        "mean over the reference read: x' = x - mean<sub>speaker</sub>. A value now reads as 'higher / louder / "
        "faster than this narrator's normal read'. Only means are used (no division by the speaker's spread), which "
        "is more robust with 50 reference clips.",
        "<b>Groups.</b> The 7 feature groups of Final_Report.pdf, with Formants split into frequency, bandwidth and "
        "amplitude (6 features each): 9 groups.",
        "<b>Group scores.</b> Inside each fold, each group is standardised and reduced by PCA, fitted on the "
        "training actors only. <i>PC1 per group</i> gives 9 inputs; <i>k PCs (90%)</i> keeps, per group, the fewest "
        "components that explain at least 90% of that group's variance. Each PC1 is oriented by an anchor "
        "feature so that '+' has a fixed meaning (e.g. Loudness: + = louder); this does not change predictions.",
        f"<b>Model.</b> Unchanged: StandardScaler + multinomial logistic regression, C = 1, seed {exp.SEED}; "
        "4 leave-one-actor-out folds plus the random 80/20 split (calibration clips removed from both).",
        "<b>Brier score.</b> Besides accuracy and macro-F1, the predicted probabilities are scored with the "
        "multiclass Brier score: for each clip, the sum over the 7 styles of (predicted probability - 1 if true "
        f"style else 0)<super>2</super>, averaged over clips. Lower is better; a uniform guess scores {UNIFORM_BRIER:.3f}.",
    ])

    E += [P("3. Results", "h1"), P("3.1 Overall", "h2")]
    rows = [["Normalisation", "Inputs", "# inputs", "Accuracy<br/>(mean +/- sd)", "Macro-F1<br/>(mean +/- sd)",
             "Brier<br/>(mean +/- sd)", "Random split<br/>acc. / Brier", "Gap random -<br/>held-out"]]
    for (n, i), r in sm.iterrows():
        rows.append([NORM_LABEL[n], INPUT_LABEL[i], f"{r.n_inputs_mean:.0f}" if i != "kpc90" else f"{r.n_inputs_mean:.1f}",
                     f"{pct(r.acc_mean)} +/- {r.acc_sd * 100:.1f}", f"{r.f1_mean:.3f} +/- {r.f1_sd:.3f}",
                     f"{r.brier_mean:.3f} +/- {r.brier_sd:.3f}", f"{pct(r.random_acc)} / {r.random_brier:.3f}",
                     f"{r.gap_random_minus_loao * 100:.1f} pts"])
    E += [table(rows, [2.1 * cm, 2.3 * cm, 1.3 * cm, 2.2 * cm, 2.3 * cm, 2.3 * cm, 2.3 * cm, W - 14.8 * cm]),
          P("Leave-one-actor-out, mean of the 4 folds. k PCs (90%): the number of inputs varies slightly per fold "
            f"(mean shown). Chance accuracy is 14.3%; Brier: lower is better, a uniform guess scores {UNIFORM_BRIER:.3f}. "
            "Gap: random-split accuracy minus held-out-actor accuracy.", "cap")]

    E += [P("3.2 Per held-out actor (accuracy)", "h2")]
    rows = [["Normalisation", "Inputs"] + SPEAKERS]
    for (n, i) in sm.index:
        rows.append([NORM_LABEL[n], INPUT_LABEL[i]] + [pct(acc_spk.loc[(n, i), s]) for s in SPEAKERS])
    E += [table(rows, [2.6 * cm, 3.0 * cm] + [(W - 5.6 * cm) / 4] * 4),
          P(f"Speaker normalisation mainly rescues ex03 ({pts(d_ex03)} points with 88 features), the hardest "
            "held-out actor without it.", "cap")]

    E += [P("3.3 Quality of the predicted probabilities (Brier score)", "h2")]
    b_spk = fm.pivot_table(index=["norm", "inputs"], columns="test_speaker", values="brier")
    rows = [["Normalisation", "Inputs"] + SPEAKERS]
    for (n, i) in sm.index:
        rows.append([NORM_LABEL[n], INPUT_LABEL[i]] + [f"{b_spk.loc[(n, i), s]:.3f}" for s in SPEAKERS])
    E += [table(rows, [2.6 * cm, 3.0 * cm] + [(W - 5.6 * cm) / 4] * 4),
          P("Multiclass Brier score per held-out actor: the mean over clips of the squared distance between the 7 "
            "predicted probabilities and the one-hot true style (0 = perfect, 2 = fully confident and wrong, "
            f"{UNIFORM_BRIER:.3f} = uniform guess). Unlike accuracy, it also rewards being unsure when wrong.", "cap")]
    b_raw3, b_spk3 = b_spk.loc[("raw", "all88"), "ex03"], b_spk.loc[("spknorm", "all88"), "ex03"]
    E += bullets([
        f"With raw features the ex03 model is <b>confidently wrong</b>: its Brier score ({b_raw3:.3f}) is no better than "
        f"a uniform guess ({UNIFORM_BRIER:.3f}), although it gets {pct(acc_spk.loc[('raw', 'all88'), 'ex03'])} right. "
        f"Speaker normalisation brings it to {b_spk3:.3f}, and the mean over actors from {base.brier_mean:.3f} to "
        f"{spk.brier_mean:.3f}.",
        f"The 9-score model has worse probabilities on average ({spc1.brier_mean:.3f}) but the most even ones across "
        f"actors (sd {spc1.brier_sd:.3f} against {spk.brier_sd:.3f} for 88 features).",
        "For the director this matters as much as accuracy: a rule such as 'flag the take when the requested style "
        "gets less than 50%' only works if the probabilities can be trusted on a new voice.",
    ])

    E += [P("3.4 Per style (recall, mean of the 4 held-out actors)", "h2")]
    cols = [("raw", "all88"), ("spknorm", "all88"), ("spknorm", "pc1"), ("spknorm", "kpc90")]
    rows = [["Style"] + [f"{NORM_LABEL[n]}<br/>{INPUT_LABEL[i]}" for n, i in cols] + ["Change from<br/>speaker-norm (88)"]]
    for s in STYLES:
        rows.append([s] + [pct(rec.loc[s, c]) for c in cols] + [pts(rec.loc[s, cols[1]] - rec.loc[s, cols[0]])])
    E += [table(rows, [2.4 * cm] + [(W - 5.2 * cm) / 4] * 4 + [2.8 * cm]),
          P("Last column: speaker-normalised minus raw, both with 88 features (percentage points).", "cap")]

    half_cm = W / cm / 2 - 0.2   # img() takes a width in cm
    figs = Table([[img(RES / "confusion_raw_all88.png", half_cm),
                   img(RES / "confusion_spknorm_all88.png", half_cm)]],
                 colWidths=[W / 2, W / 2])
    E += [KeepTogether([figs, P("Confusion matrices summed over the 4 held-out actors: raw (left) and "
                                "speaker-normalised (right), 88 features.", "cap")])]

    E += [P("4. Speaker normalisation", "h1"), P("4.1 What changes", "h2")]
    def pred_ratio(name):  # predicted count / true count per style, summed over the 4 held-out actors
        c = pd.read_csv(RES / f"confusion_{name}.csv", index_col=0)
        return c.sum(axis=0) / c.sum(axis=1)
    ratio_raw, ratio_spk = pred_ratio("raw_all88"), pred_ratio("spknorm_all88")
    shifts = (rec[("spknorm", "all88")] - rec[("raw", "all88")]).sort_values()
    up = ", ".join(f"{s} {pts(v)}" for s, v in shifts[::-1].items() if v > 0.02)
    down = ", ".join(f"{s} {pts(v)}" for s, v in shifts.items() if v < -0.02)
    E += bullets([
        f"Mean accuracy {pts(spk.acc_mean - base.acc_mean)} points and macro-F1 {spk.f1_mean - base.f1_mean:+.3f}; "
        f"the spread across actors falls from {base.acc_sd * 100:.1f} to {spk.acc_sd * 100:.1f} points, so the result "
        "depends much less on which actor is held out.",
        f"Recall moves between styles rather than rising everywhere. Up: {up}. Down: {down}.",
        f"The catch-all label changes. Without normalisation the model predicts confused {ratio_raw['confused']:.2f}x "
        f"and sad {ratio_raw['sad']:.2f}x as often as they occur; with it, <b>default</b> is predicted "
        f"{ratio_spk['default']:.2f}x as often (laughing {ratio_spk['laughing']:.2f}x, confused "
        f"{ratio_spk['confused']:.2f}x). Clips that sit close to the narrator's neutral read are called default "
        "(Section 4.2).",
        f"The random-split gap shrinks only a little ({base.gap_random_minus_loao * 100:.1f} to "
        f"{spk.gap_random_minus_loao * 100:.1f} points): a single mean shift per actor does not remove everything "
        "the model learns about a known voice.",
    ])
    E += [P("4.2 Caveat: the reference read is default speech", "h2"),
          P(f"Because each actor's mean is taken from default clips, a held-out default clip lands near zero by "
            f"construction, so default is easier to recognise ({pts(d_def)} points recall). The reference clips are "
            "never tested on, so this is not leakage, but part of the gain belongs to the design: the model is "
            "answering 'how does this take differ from the narrator's neutral read?'. That matches the audiobook "
            "director's question, and it must be stated when the numbers are compared with the ResNet, which has no "
            "reference read.")]

    E += [P("5. One score per feature group", "h1"), P("5.1 How much of each group one score captures", "h2")]
    p1 = pg[(pg.norm == "spknorm") & (pg.inputs == "pc1")].groupby("group", sort=False)
    pk = pg[(pg.norm == "spknorm") & (pg.inputs == "kpc90")].groupby("group", sort=False)
    rows = [["Group", "Changeable?", "Features", "Variance in PC1", "PCs for 90%", "'+' direction of PC1 (anchor)"]]
    for g in exp.GROUPS9:
        a = p1.get_group(g)
        rows.append([g.replace("&", "&amp;"), exp.CHANGEABLE9[g], f"{int(a.n_features.iloc[0])}", pct(a.evr_pc1.mean()),
                     f"{pk.get_group(g).k.mean():.2g}", exp.ANCHOR[g][1]])
    E += [table(rows, [3.0 * cm, 2.5 * cm, 1.6 * cm, 2.0 * cm, 1.9 * cm, W - 11.0 * cm], shade_rows=()),
          P("Speaker-normalised features, mean of the 4 folds. PC1 directions are almost identical across folds "
            f"(|cosine| with fold 1 at least {pg[(pg.inputs == 'pc1')]['pc1_abs_cos_vs_fold1_min'].min():.3f} for every "
            "group).", "cap")]
    lo = p1.evr_pc1.mean().sort_values()
    E += bullets([
        f"Pitch level ({pct(lo['Pitch level'])}) and formant amplitude ({pct(lo['Formant amplitude'])}) are essentially "
        "one number each, so collapsing them loses nothing.",
        f"Spectral / timbre ({pct(lo['Spectral / timbre'])} in PC1, {pk.get_group('Spectral / timbre').k.mean():.0f} PCs "
        f"for 90%) and voice quality ({pct(lo['Voice quality'])}) are not one-dimensional; most of the accuracy lost "
        "with 9 inputs is lost here.",
        f"Accuracy with speaker normalisation: 9 inputs {pct(spc1.acc_mean)}, k PCs {pct(skpc.acc_mean)} "
        f"(about {skpc.n_inputs_mean:.0f} inputs), 88 features {pct(spk.acc_mean)}. Without normalisation the 9-input "
        f"model reaches {pct(rpc1.acc_mean)}.",
    ])

    E += [P("5.2 What the 9-score model uses", "h2")]
    c = co[co.norm == "spknorm"]
    cm_ = c.pivot_table(index="style", columns="group", values="coef", aggfunc="mean")[exp.GROUPS9]
    cs_ = c.pivot_table(index="style", columns="group", values="coef", aggfunc="std")[exp.GROUPS9]
    short = {"Rate & pauses": "Rate &amp;<br/>pauses", "Pitch movement": "Pitch<br/>movement", "Pitch level": "Pitch<br/>level",
             "Voice quality": "Voice<br/>quality", "Spectral / timbre": "Spectral /<br/>timbre",
             "Formant frequency": "Formant<br/>freq.", "Formant bandwidth": "Formant<br/>bandw.",
             "Formant amplitude": "Formant<br/>ampl."}
    rows = [["Style"] + [short.get(g, g) for g in exp.GROUPS9]]
    for s in STYLES:
        rows.append([s] + [f"{cm_.loc[s, g]:+.2f}" for g in exp.GROUPS9])
    E += [table(rows, [2.0 * cm] + [(W - 2.0 * cm) / 9] * 9, shade_rows=(1, 2)),
          P("Standardised coefficients of the speaker-normalised 9-score model, mean of the 4 folds. Shaded: groups a "
            "narrator can change. '+' pushes towards the style when the group score is above the narrator's normal "
            f"read in the anchor direction (Section 5.1). Largest sd of a coefficient across folds: {cs_.values.max():.2f}.",
            "cap")]
    E += bullets([
        "After normalisation, pitch level means 'higher than this narrator's own neutral read', so it no longer "
        "encodes voice pitch (the speaker-sex proxy) directly. It is now the strongest cue for happy "
        f"({cm_.loc['happy', 'Pitch level']:+.2f}) and against sad ({cm_.loc['sad', 'Pitch level']:+.2f}).",
        f"Rate &amp; pauses separates confused ({cm_.loc['confused', 'Rate & pauses']:+.2f}) and enunciated "
        f"({cm_.loc['enunciated', 'Rate & pauses']:+.2f}) from whisper ({cm_.loc['whisper', 'Rate & pauses']:+.2f}) and "
        f"default ({cm_.loc['default', 'Rate & pauses']:+.2f}); loudness has small weights for every style (|coef| "
        f"at most {cm_['Loudness'].abs().max():.2f}).",
        "Read with care: '+' on rate &amp; pauses means more voiced segments per second, but this PC1 holds only about "
        f"{pct(p1.get_group('Rate & pauses').evr_pc1.mean())} of the group, so it is not a pure speaking-rate dial.",
    ])

    E += [P("5.3 Stability of the coefficients across folds", "h2")]
    rows = [["Normalisation", "Inputs", "Mean pairwise Spearman (4 folds)"]]
    for (n, i), r in st.iterrows():
        rows.append([NORM_LABEL[n], INPUT_LABEL[i], f"{r.coef_spearman_mean_pairwise:.3f}"])
    E += [table(rows, [3 * cm, 3.5 * cm, 6 * cm]),
          P("Spearman correlation between the 7 x p coefficient matrices of every pair of fold models (6 pairs), "
            "averaged. Each fold has a different held-out actor and so different training data.", "cap")]

    E += [P("6. How much speaker identity is left in the inputs", "h1")]
    rows = [["Normalisation", "Inputs", "Speaker-ID accuracy (5-fold CV)"]]
    for (n, i), r in pr.iterrows():
        rows.append([NORM_LABEL[n], INPUT_LABEL[i], f"{pct(r.speaker_id_accuracy)} +/- {r.sd * 100:.1f}"])
    E += [table(rows, [3 * cm, 3.5 * cm, 6 * cm]),
          P("A logistic regression trained to name the actor (4 classes, chance 25%) from the same inputs, on all "
            "evaluated clips. Lower means the inputs carry less information about who is speaking.", "cap")]
    E += bullets([
        "Mean-only normalisation removes the speaker's average but not the speaker's spread or feature correlations, "
        "so the actor is still easy to recognise from 88 features.",
        "Fewer, coarser inputs carry less identity: the 9 group scores are the least identifying input, which is "
        "also why they are less accurate.",
    ])

    E += [P("7. Limitations and open decisions", "h1")]
    E += bullets([
        "Four actors: each fold rests on one held-out voice, and the ex03 gain carries most of the average improvement.",
        "Only the default-reference, mean-only variant was run (team choice); dividing by the speaker's spread, or a "
        "reference of random clips, was not tested.",
        "The comparison with the ResNet is fair only if it gets the same reference read (e.g. subtracting the "
        "speaker's mean log-mel spectrum) or if this difference is reported alongside the numbers.",
        "Open decision: which configuration becomes the official baseline, and whether the 9-score model is used for "
        "the counterfactual and human-study explanations (more stable and readable, about "
        f"{(spk.acc_mean - spc1.acc_mean) * 100:.0f} points less accurate).",
    ])
    E += [Spacer(1, 6), P("Reproduce: scripts/07_experiments_spknorm_pca.py (results/experiments/, splits/calibration.csv), "
                          "then scripts/08_build_experiments_report.py.", "cap")]

    out.parent.mkdir(exist_ok=True)
    doc.build(E, onFirstPage=footer, onLaterPages=footer)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
