"""Build the PDF report (model training, results, explanation mechanism) from results/.

Run after 03_train_logreg.py and 04_group_contributions.py.
Needs reportlab (not part of the pipeline requirements):
    pip install reportlab
Usage:
    python scripts/05_build_report.py [--out report/Report.pdf]   (run from the "Expresso Project" folder)
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_groups import CHANGEABLE, DESCRIPTION, GROUP_NAMES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = ROOT / "report" / "Report.pdf"

STYLES = ["confused", "default", "enunciated", "happy", "laughing", "sad", "whisper"]
SPEAKERS = ["ex01", "ex02", "ex03", "ex04"]
INK, INK2, MUTED = colors.HexColor("#0b0b0b"), colors.HexColor("#52514e"), colors.HexColor("#898781")
RULE, HEAD_BG, ACCENT = colors.HexColor("#e1e0d9"), colors.HexColor("#f0efec"), colors.HexColor("#1c5cab")
SIGN_EPS = 0.1  # |mean signed contribution| below this (logits) is shown as ~0

ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("t", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=19, leading=23,
                            alignment=TA_LEFT, textColor=INK, spaceAfter=4),
    "sub": ParagraphStyle("s", fontName="Helvetica", fontSize=10, leading=14, textColor=INK2),
    "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=13.5, leading=17, textColor=ACCENT,
                         spaceBefore=14, spaceAfter=6, keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=10.5, leading=14, textColor=INK,
                         spaceBefore=8, spaceAfter=3, keepWithNext=1),
    "body": ParagraphStyle("b", fontName="Helvetica", fontSize=9.5, leading=13.5, textColor=INK, spaceAfter=5),
    "bullet": ParagraphStyle("bl", fontName="Helvetica", fontSize=9.5, leading=13.5, textColor=INK,
                             leftIndent=12, bulletIndent=2, spaceAfter=2),
    "cell": ParagraphStyle("c", fontName="Helvetica", fontSize=8.3, leading=10.5, textColor=INK),
    "cellb": ParagraphStyle("cb", fontName="Helvetica-Bold", fontSize=8.3, leading=10.5, textColor=INK),
    "cap": ParagraphStyle("cap", fontName="Helvetica", fontSize=8, leading=10.5, textColor=INK2, spaceAfter=8),
}


def P(text, style="body"):
    return Paragraph(text, S[style])


def bullets(items):
    return [Paragraph(t, S["bullet"], bulletText="•") for t in items]


def table(rows, widths, header=True, bold_last=False, shade_rows=()):
    data = [[c if not isinstance(c, str) else Paragraph(c, S["cellb"] if (header and i == 0) else S["cell"])
             for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    st = [
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), HEAD_BG), ("LINEBELOW", (0, 0), (-1, 0), 0.8, MUTED)]
    if bold_last:
        st += [("BACKGROUND", (0, -1), (-1, -1), HEAD_BG)]
    for c in shade_rows:  # shade whole columns (e.g. the changeable groups)
        st += [("BACKGROUND", (c, 1), (c, -1), colors.HexColor("#eef4fc"))]
    t.setStyle(TableStyle(st))
    return t


def img(path, width_cm):
    from reportlab.lib.utils import ImageReader
    w, h = ImageReader(str(path)).getSize()
    return Image(str(path), width=width_cm * cm, height=width_cm * cm * h / w)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(2 * cm, 1.2 * cm, "Black Box group - eGeMAPS + logistic regression baseline")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


def pct(x):
    return f"{x * 100:.1f}%"


def sign(v):
    return "+" if v >= SIGN_EPS else ("-" if v <= -SIGN_EPS else "~0")


def group_table(share, signed, W):
    """Rows = style, columns = 7 groups (+ changeable total); cell = share % and direction."""
    short = {"Rate & pauses": "Rate &amp; pauses", "Spectral / timbre": "Spectral / timbre"}
    head = ["Style"] + [f"{short.get(g, g)}<br/><font size='6.5' color='#52514e'>{CHANGEABLE[g]}</font>"
                        for g in GROUP_NAMES] + ["Change-<br/>able total<br/><font size='6.5' color='#52514e'>rate + loudness</font>"]
    rows = [head]
    for s in STYLES:
        cells = [f"{share.loc[s, g]:.1f}% <font color='#52514e'>{sign(signed.loc[s, g])}</font>" for g in GROUP_NAMES]
        changeable = share.loc[s, ["Rate & pauses", "Loudness"]].sum()
        rows.append([s] + cells + [f"<b>{changeable:.1f}%</b>"])
    first, last = 1.8 * cm, 1.9 * cm
    rest = (W - first - last) / len(GROUP_NAMES)
    return table(rows, [first] + [rest] * len(GROUP_NAMES) + [last], shade_rows=(1, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    out = ap.parse_args().out

    clips = pd.read_csv(ROOT / "clips.csv")
    folds = pd.read_csv(ROOT / "splits" / "folds.csv")
    m = pd.read_csv(RES / "metrics_per_fold.csv")
    ps = pd.read_csv(RES / "per_style.csv")
    sens = json.loads((RES / "c_sensitivity_macro_f1.json").read_text())
    cmpc = pd.read_csv(RES / "c_tuned_vs_fixed.csv")
    fg = pd.read_csv(RES / "feature_groups.csv")
    gl = pd.read_csv(RES / "group_contributions_loao.csv")
    gr = pd.read_csv(RES / "group_contributions_random.csv")
    cmr = pd.read_csv(RES / "confusion_all_folds.csv", index_col=0)
    cmr = cmr.div(cmr.sum(axis=1), axis=0)

    fm = m[m["split"].str.startswith("fold")]
    mean = m[m["split"] == "mean_4folds"].iloc[0]
    sd = m[m["split"] == "std_4folds"].iloc[0]
    rnd = m[m["split"] == "random_80_20"].iloc[0]
    gap = rnd["accuracy"] - mean["accuracy"]
    worst = fm.loc[fm["macro_f1"].idxmin()]
    wk = worst["split"].replace("fold", "")
    cw = pd.read_csv(RES / f"confusion_fold{wk}.csv", index_col=0)
    cw = cw.div(cw.sum(axis=1), axis=0)
    loao = ps[ps["split"].str.startswith("fold")]
    f1_style = loao.groupby("style")["f1"].mean()
    rec = loao.pivot(index="style", columns="test_speaker", values="recall")

    share_l = gl.pivot(index="style", columns="group", values="share_mean")[GROUP_NAMES]
    sign_l = gl.pivot(index="style", columns="group", values="signed_mean")[GROUP_NAMES]
    sd_l = gl.pivot(index="style", columns="group", values="share_sd")[GROUP_NAMES]
    share_r = gr.pivot(index="style", columns="group", values="share")[GROUP_NAMES]
    sign_r = gr.pivot(index="style", columns="group", values="signed")[GROUP_NAMES]
    n_feat = fg["group"].value_counts().reindex(GROUP_NAMES)
    chg_l = share_l[["Rate & pauses", "Loudness"]].sum(axis=1)
    chg_r = share_r[["Rate & pauses", "Loudness"]].sum(axis=1)
    per_feat = share_l.mean().div(n_feat)  # average share per feature, over styles
    diff = (share_r - share_l)
    big = diff.abs().stack().sort_values(ascending=False).head(3)

    W = A4[0] - 4 * cm
    story = []

    # ---------- Title + summary ----------
    story += [
        P("Speech style classification with eGeMAPS + logistic regression", "title"),
        P("Report for the Black Box group (Explainable Data Science, AIT) - glass-box baseline path: "
          "model training, results and explanation mechanism<br/>"
          "Prepared by Aung Khant Maung - 12 November 2026", "sub"),
        Spacer(1, 10),
        P("Summary", "h1"),
        P(f"We trained a logistic regression on the 88 eGeMAPS acoustic features of 10,636 Expresso read-speech "
          f"clips (4 actors, 7 styles), with the regularisation strength fixed at C = 1. Evaluated by leaving one "
          f"whole actor out, it reaches <b>{pct(mean['accuracy'])} accuracy</b> (macro-F1 "
          f"<b>{mean['macro_f1']:.3f}</b>, mean of 4 folds; chance is 14.3%). On a random split where every actor "
          f"is seen in training it reaches {pct(rnd['accuracy'])}: a <b>{gap * 100:.0f}-point gap</b>, so the "
          f"features carry a lot of speaker-specific information."),
    ]
    story += bullets([
        f"Whisper is almost always recognised (mean F1 {f1_style['whisper']:.2f}); happy "
        f"({f1_style['happy']:.2f}) and default ({f1_style['default']:.2f}) are the hardest styles. Macro-F1 ranges "
        f"from {fm['macro_f1'].min():.2f} ({worst['test_speaker']}) to {fm['macro_f1'].max():.2f} across held-out actors.",
        "Tuning C per fold instead of fixing it changes accuracy by at most "
        f"{(cmpc['accuracy'] - cmpc['accuracy_fixed_C']).abs().max() * 100:.1f} points; C = 1 keeps the explanations "
        "comparable across folds.",
        f"Explanation mechanism: the features a narrator can change (rate &amp; pauses, loudness) carry "
        f"{chg_l.min():.0f}-{chg_l.max():.0f}% of each style's contribution; most of the model's evidence is "
        f"spectral / timbre and formant features, which a narrator cannot change.",
    ])

    # ---------- 1. Data, features, splits ----------
    ct = clips.groupby(["style", "speaker_id"]).size()
    story += [P("1. Data, features and splits", "h1"),
              P(f"<b>Data.</b> Expresso read speech (Hugging Face <font face='Courier'>ylacombe/expresso</font>, "
                f"config <font face='Courier'>read</font>, CC BY-NC 4.0): 4 professional actors (ex01-ex04), "
                f"{len(clips):,} clips in 7 styles (confused, default, enunciated, happy, laughing, sad, whisper), "
                f"{ct.min()}-{ct.max()} clips per style per actor, {clips['duration_s'].sum() / 3600:.1f} hours. The "
                "set also holds emphasis, essentials and longform, which the official README calls 'default "
                "substyles', and singing; all four were dropped (team decision). The dataset gives only '2 males, "
                "2 females' with no mapping to actor ids, so results are reported per actor."),
              P("<b>Features.</b> Each clip is resampled from 48 to 16 kHz, kept at full length (no crop or pad, "
                "which would distort pause and loudness statistics) and not loudness-normalised (loudness must "
                "stay usable). openSMILE eGeMAPSv02 functionals give 88 features per clip: pitch, loudness, "
                "spectral balance, formants, voice quality and timing."),
              P(f"<b>Splits.</b> Main evaluation is leave-one-actor-out: 4 folds, each training on 3 actors "
                f"(about {len(folds) - folds.groupby('fold').size().max():,} clips) and testing on all 7 styles of "
                f"the fourth (about {folds.groupby('fold').size().min():,} clips); fold k holds out ex0k. For "
                "comparison, a random 80/20 split stratified by style (seed 42) mixes all actors in training and "
                "test, so the gap between the two shows how much the model relies on knowing the speaker.")]

    # ---------- 2. Model training ----------
    story += [P("2. Model training", "h1")]
    story += bullets([
        "<b>Pipeline:</b> StandardScaler, then multinomial logistic regression (lbfgs, max_iter 5000, no class "
        "weights because the classes are balanced, seed 42). One model per fold, plus one for the random split.",
        "<b>No leakage:</b> the scaler and the model are fit on the training actors only; the test actor is "
        "never used for scaling, training or any choice of settings.",
        "<b>C = 1 in every fold.</b> The regularisation strength is fixed so that coefficients, and the "
        "explanations built from them (Section 4), are on the same scale in every fold.",
        "Training all models takes about 10 seconds on a laptop CPU, with no convergence warnings.",
    ])
    grid = list(sens["fold1"].keys())
    rows = [["Fold"] + [f"C={c}" for c in grid]]
    for k in range(1, 5):
        sc = sens[f"fold{k}"]
        best = max(sc, key=sc.get)
        rows.append([f"{k}"] + [f"<b>{v:.3f}</b>" if c == best else f"{v:.3f}" for c, v in sc.items()])
    within = max(max(sens[f"fold{k}"].values()) - sens[f"fold{k}"]["1"] for k in range(1, 5))
    story += [P("2.1 Sensitivity check: does C matter?", "h2"),
              P("Every C in {0.001, 0.01, 0.1, 1, 10, 100} was scored by an inner leave-one-actor-out over the 3 "
                "training actors of each fold (train on 2, macro-F1 on the 3rd, rotate 3 times). The scores are "
                "only reported; they do not choose C."),
              table(rows, [1.4 * cm] + [(W - 1.4 * cm) / len(grid)] * len(grid)),
              P(f"Inner macro-F1 on the training actors; best value per fold in bold. From C = 0.01 to 100 the "
                f"curve is flat (changes under 0.02), and C = 1 is within {within:.3f} of the best value in every "
                "fold. With about 8,000 clips and 88 features, regularisation has little to do.", "cap")]

    rows = [["Evaluation", "C picked by the scan", "Accuracy (picked C)", "Accuracy (C = 1)",
             "Macro-F1 (picked C)", "Macro-F1 (C = 1)"]]
    for _, r in cmpc.iterrows():
        name = r["split"].replace("fold", "Fold ").replace("random_80_20", "Random 80/20 split")
        rows.append([name, f"{r['tuned_C']:g}", pct(r["accuracy"]), pct(r["accuracy_fixed_C"]),
                     f"{r['macro_f1']:.3f}", f"{r['macro_f1_fixed_C']:.3f}"])
    fl = cmpc[cmpc["split"].str.startswith("fold")]
    rows.append(["<b>Mean, 4 folds</b>", "", f"<b>{pct(fl['accuracy'].mean())}</b>",
                 f"<b>{pct(fl['accuracy_fixed_C'].mean())}</b>", f"<b>{fl['macro_f1'].mean():.3f}</b>",
                 f"<b>{fl['macro_f1_fixed_C'].mean():.3f}</b>"])
    story += [P("2.2 Tuned C per fold vs fixed C = 1", "h2"),
              P("An earlier version of this work let each fold use the C that scored best in the scan above. "
                "Because the scores are almost tied, the picks were scattered (10, 0.1, 1, 100). The table "
                "compares both choices on the same test actors."),
              table(rows, [3.4 * cm, 2.8 * cm] + [(W - 6.2 * cm) / 4] * 4, bold_last=False),
              P("The fixed C costs at most "
                f"{(cmpc['accuracy'] - cmpc['accuracy_fixed_C']).max() * 100:.1f} accuracy points in any evaluation "
                f"(at most {(fl['accuracy'] - fl['accuracy_fixed_C']).abs().max() * 100:.1f} in a held-out-actor fold) and "
                "the 4-fold means are practically identical, so all results below use C = 1.", "cap")]

    # ---------- 3. Results ----------
    story += [P("3. Results", "h1"), P("3.1 Overall", "h2")]
    rows = [["Evaluation", "Test actor", "Train / test clips", "Accuracy", "Macro-F1"]]
    for _, r in fm.iterrows():
        rows.append([r["split"].replace("fold", "Fold "), r["test_speaker"],
                     f"{int(r['n_train']):,} / {int(r['n_test']):,}", pct(r["accuracy"]), f"{r['macro_f1']:.3f}"])
    rows.append(["<b>Mean +/- sd, 4 folds</b>", "", "", f"<b>{pct(mean['accuracy'])} +/- {sd['accuracy'] * 100:.1f}</b>",
                 f"<b>{mean['macro_f1']:.3f} +/- {sd['macro_f1']:.3f}</b>"])
    rows.append(["Random 80/20 split", "all mixed", f"{int(rnd['n_train']):,} / {int(rnd['n_test']):,}",
                 pct(rnd["accuracy"]), f"{rnd['macro_f1']:.3f}"])
    story.append(table(rows, [4 * cm, 2.4 * cm, 3.4 * cm, 3.6 * cm, W - 13.4 * cm]))
    story += [Spacer(1, 4),
              P(f"Chance level is 14.3%. The random split scores {gap * 100:.0f} points higher, so much of what "
                "the model learns is tied to the individual speaker; the leave-one-actor-out numbers are the "
                "baseline to report."),
              P("3.2 Per style", "h2")]
    rs = ps[ps["split"] == "random_80_20"].set_index("style")["f1"]
    rows = [["Style"] + [f"Recall {s}" for s in SPEAKERS] + ["Mean F1 (4 folds)", "F1 random split"]]
    for s in STYLES:
        rows.append([s] + [pct(rec.loc[s, c]) for c in SPEAKERS] + [f"{f1_style[s]:.3f}", f"{rs[s]:.3f}"])
    story.append(table(rows, [2.3 * cm] + [2 * cm] * 4 + [2.6 * cm, W - 12.9 * cm]))
    story += [P("Recall = share of that actor's clips of the style labelled correctly (about 380 clips per cell).",
                "cap")]
    story += bullets([
        f"<b>Whisper</b> is recognised {rec.loc['whisper'].min() * 100:.0f}-{rec.loc['whisper'].max() * 100:.0f}% "
        "of the time for every actor. It is the easy class the proposal warned would inflate overall accuracy, "
        "so per-style results are always reported.",
        "<b>Laughing</b> and <b>sad</b> are next best. <b>Happy</b> and <b>default</b> are the weakest; happy is "
        "mostly mistaken for laughing, and default is spread over sad, confused and happy.",
        f"Per-actor differences are large: enunciated is {rec.loc['enunciated', 'ex01'] * 100:.0f}% correct for "
        f"ex01 but {rec.loc['enunciated', 'ex03'] * 100:.0f}% for ex03; happy is {rec.loc['happy', 'ex04'] * 100:.0f}% "
        f"for ex04 but {rec.loc['happy', 'ex03'] * 100:.0f}% for ex03.",
    ])
    story += [KeepTogether([P("3.3 Confusion matrices", "h2"), img(RES / "confusion_all_folds.png", 9.0),
                            P("Summed over the 4 held-out actors (each clip is tested exactly once). Rows are the "
                              "true style, columns the prediction; cells show the share of the row and the count.",
                              "cap")]),
              P("Main confusions: confused, enunciated and sad blur together "
                f"({cmr.loc['enunciated', 'confused']:.0%} of enunciated and {cmr.loc['sad', 'confused']:.0%} of "
                f"sad clips are called confused); happy goes to laughing ({cmr.loc['happy', 'laughing']:.0%}) and "
                f"default ({cmr.loc['happy', 'default']:.0%}); default has no clear home "
                f"({cmr.loc['default', 'default']:.0%} correct).")]
    grid_imgs = [[img(RES / f"confusion_fold{k}.png", 6.7) for k in (1, 2)],
                 [img(RES / f"confusion_fold{k}.png", 6.7) for k in (3, 4)]]
    gt = Table(grid_imgs, colWidths=[7.0 * cm] * 2, hAlign="CENTER")
    gt.setStyle(TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("RIGHTPADDING", (0, 0), (-1, -1), 0)]))
    story += [P("Per fold", "h2"), gt,
              P(f"Fold {wk} (test actor {worst['test_speaker']}) is the hardest (macro-F1 {worst['macro_f1']:.2f}): "
                f"its default clips are labelled sad or confused ({cw.loc['default', 'default']:.0%} correct) and "
                f"{cw.loc['happy', 'laughing']:.0%} of its happy clips are labelled laughing. Full-size images are "
                "in results/.", "cap")]

    # ---------- 4. Explanation mechanism ----------
    story += [P("4. Explanation mechanism", "h1"),
              P("The model is a glass box: its score (logit) for a style is a sum of one term per feature. For a "
                "clip x, the contribution of feature f to style s is <b>coef[s, f] x (x<sub>f</sub> - mean<sub>f"
                "</sub>) / sd<sub>f</sub></b>, with the mean and standard deviation taken from the training actors "
                "of that fold. These contributions add up exactly to the logit (minus the intercept), and for a "
                "linear model they are also the SHAP values, so no approximation is involved. They are the "
                "reference explanations for checking LIME and Integrated Gradients later.")]

    coef = sum(pd.read_csv(RES / f"coefficients_fold{k}.csv", index_col=0) for k in range(1, 5)) / 4
    rows = [["Style", "Top 3 features raising this style (mean standardised coefficient, 4 folds)"]]
    for s in STYLES:
        top = coef.loc[s].sort_values(ascending=False).head(3)
        rows.append([s, "<br/>".join(f"<font face='Courier' size='7.5'>{n}</font> ({v:+.2f})" for n, v in top.items())])
    story += [P("4.1 Strongest individual features", "h2"), table(rows, [2.3 * cm, W - 2.3 * cm]),
              P("Many eGeMAPS features are strongly correlated, so weight can shift between near-duplicates. "
                "The top features changed noticeably when C changed (Section 2.2), so single features should not "
                "be over-read; the groups below summarise families of related features instead.", "cap")]

    rows = [["Group", "Features", "Changeable by the narrator?", "What it contains"]]
    desc = DESCRIPTION
    for g in GROUP_NAMES:
        rows.append([g.replace("&", "&amp;"), str(int(n_feat[g])), CHANGEABLE[g], desc[g]])
    story += [P("4.2 Feature groups for actionable recourse", "h2"),
              P("For the audiobook director, an explanation is only useful if it points at something the narrator "
                "can change on the next take. Following proposal Section 6.4, the 88 features are grouped into 7 "
                "families of correlated features and labelled by whether a narrator can change them (speaking "
                "rate, pauses and loudness can; timbre, pitch level and vocal-tract shape cannot)."),
              table(rows, [2.9 * cm, 1.6 * cm, 3.2 * cm, W - 7.7 * cm])]

    note = ("Each cell: share of the style's total absolute contribution that comes from the group, averaged over "
            "the test clips of that true style (rows sum to 100%), then the average direction: + the group pushes "
            "the clip toward its style, - away from it, ~0 small (|mean| under 0.1 logit). Shaded columns are the "
            "groups a narrator can change.")
    story += [KeepTogether([
        P("4.3 Group contributions per style, leave-one-actor-out (mean of 4 folds)", "h2"),
        group_table(share_l, sign_l, W),
        P(note + f" Standard deviation of a share across the 4 folds is at most {sd_l.values.max():.1f} points.",
          "cap")])]
    story += [KeepTogether([
        P("4.4 Group contributions per style, random-split model (for comparison)", "h2"),
        group_table(share_r, sign_r, W),
        P(note + " Model trained on all 4 actors, explained on its 2,128 test clips.", "cap")])]

    big_txt = "; ".join(f"{s} / {g.replace('&', '&amp;')} {share_l.loc[s, g]:.1f}% to {share_r.loc[s, g]:.1f}%"
                        for (s, g), _ in big.items())
    story += [P("What the explanations say", "h2")]
    story += bullets([
        f"<b>Little of the evidence is changeable.</b> Rate &amp; pauses plus loudness account for "
        f"{chg_l.min():.0f}-{chg_l.max():.0f}% of the contribution per style (highest for "
        f"{chg_l.idxmax()}, {chg_l.max():.0f}%; lowest for {chg_l.idxmin()}, {chg_l.min():.0f}%). Most of the model's "
        "evidence comes from spectral / timbre and formant features, which describe the voice itself. A recourse "
        "message such as 'speak louder' or 'slow down' can therefore only move a small part of the score.",
        f"<b>Whisper</b> is the exception: rate &amp; pauses ({share_l.loc['whisper', 'Rate & pauses']:.0f}%, "
        f"{sign(sign_l.loc['whisper', 'Rate & pauses'])}) and voice quality "
        f"({share_l.loc['whisper', 'Voice quality']:.0f}%, {sign(sign_l.loc['whisper', 'Voice quality'])}) matter far "
        "more than for any other style, matching the absence of voiced (pitched) speech in a whisper.",
        f"<b>Sad</b> relies most on pitch: pitch movement {share_l.loc['sad', 'Pitch movement']:.0f}% and pitch level "
        f"{share_l.loc['sad', 'Pitch level']:.0f}%, about twice the other styles. Pitch level is the sex proxy "
        "listed in the proposal's ethics section, so this is where speaker-dependent bias is most likely.",
        "<b>Group size does not explain it.</b> A group with more features can collect a larger share, so "
        "the shares were also divided by the number of features. Per feature, the average share (over styles) is "
        + ", ".join(f"{g.replace('&', '&amp;')} {per_feat[g]:.2f}%" for g in per_feat.sort_values(ascending=False).index)
        + ". Spectral / timbre and formant features are individually as strong as any, so their dominance is "
        "not only a counting effect.",
        f"<b>Random split vs held-out actor.</b> The group pattern is similar (no share moves by more than "
        f"{big.iloc[0]:.1f} points); the largest shifts are {big_txt}. Seeing every actor in training raises "
        f"accuracy by {gap * 100:.0f} points without changing much which kinds of evidence the model uses.",
    ])

    # ---------- 5. Limitations ----------
    story += [P("5. Limitations", "h1")]
    story += bullets([
        f"<b>Speaker dependence.</b> The {gap * 100:.0f}-point gap to the random split and the spread across folds "
        "show the features vary a lot between actors. With only 4 actors, each fold's result rests on one voice.",
        "<b>Whisper inflates accuracy.</b> Always quote per-style results next to the overall number.",
        "<b>Contributions are not causes.</b> They describe what this linear model adds up, not what makes a "
        "listener hear a style. Correlated features share credit, and a group's share also depends on how many "
        "features it contains.",
        "<b>Recourse labels are a judgement.</b> 'Changeable' follows the proposal (rate, pauses, loudness); "
        "pitch movement and voice quality are only partly under a narrator's control.",
        "<b>Acted speech, 4 North American English actors.</b> Findings may not transfer to other voices or to "
        "natural speech; per-sex analysis is not possible because sex is not given per actor.",
        "<b>Folds vs the official split.</b> Our folds follow speaker_id; they must be checked against the other "
        "team's split CSV clip for clip (see README).",
    ])

    out.parent.mkdir(exist_ok=True)
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                            bottomMargin=1.8 * cm, title="eGeMAPS + logistic regression: training, results, explanations",
                            author="Aung Khant Maung (Black Box group)")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
