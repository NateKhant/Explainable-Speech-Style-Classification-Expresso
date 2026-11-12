"""One-off reference PDF: the 7 feature groups and the eGeMAPS features in each.

Built from scripts/feature_groups.py and results/feature_groups.csv (written by
04_group_contributions.py), so it always matches the grouping used in the analysis.
Needs reportlab:  pip install reportlab
Usage:
    python scripts/06_build_feature_groups_pdf.py [--out report/Feature_Groups.pdf]
"""

import argparse
import importlib
import sys
from pathlib import Path

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Spacer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_groups import CHANGEABLE, DESCRIPTION, GROUP_NAMES  # noqa: E402

rb = importlib.import_module("05_build_report")  # shared styles and table helper
P, table, footer = rb.P, rb.table, rb.footer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report" / "Feature_Groups.pdf"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    out = ap.parse_args().out

    fg = pd.read_csv(ROOT / "results" / "feature_groups.csv")
    assert len(fg) == 88 and fg["feature"].is_unique
    W = A4[0] - 4 * cm

    story = [
        P("eGeMAPS feature groups for actionable recourse", "title"),
        P("Black Box group - eGeMAPS + logistic regression baseline. The 88 eGeMAPSv02 functionals grouped into 7 "
          "families of correlated features, labelled by whether a narrator can change them on the next take "
          "(proposal Section 6.4).", "sub"),
        Spacer(1, 8),
        P("1. Groups", "h1"),
    ]
    rows = [["Group", "Features", "Changeable by the narrator?", "What it contains"]]
    for g in GROUP_NAMES:
        rows.append([g.replace("&", "&amp;"), str(int((fg["group"] == g).sum())), CHANGEABLE[g], DESCRIPTION[g]])
    story += [table(rows, [2.9 * cm, 1.6 * cm, 3.2 * cm, W - 7.7 * cm]),
              P("Changeable follows the proposal: a narrator can change speaking rate, pauses and loudness, but not "
                "their timbre, pitch level (a proxy for speaker sex) or vocal-tract shape. Pitch movement and voice "
                "quality are only partly under a narrator's control.", "cap")]

    story += [P("2. Feature labels in each group", "h1"),
              P("Exact openSMILE eGeMAPSv02 column names, in the order they appear in "
                "<font face='Courier'>features/egemaps.parquet</font>. Every feature belongs to exactly one group "
                f"({len(fg)} features in total).")]
    rows = [["Group", "Feature labels"]]
    for g in GROUP_NAMES:
        names = fg.loc[fg["group"] == g, "feature"].tolist()
        rows.append([f"{g.replace('&', '&amp;')}<br/><font size='7' color='#52514e'>{len(names)} features"
                     f" - {CHANGEABLE[g]}</font>",
                     ", ".join(f"<font face='Courier' size='7.3'>{n}</font>" for n in names)])
    story.append(table(rows, [3.2 * cm, W - 3.2 * cm]))
    story.append(P("Naming: _sma3 = smoothed over 3 frames; nz = voiced frames only; V / UV = voiced / unvoiced "
                   "regions; amean = mean; stddevNorm = coefficient of variation; percentileXX / pctlrange0-2 = "
                   "percentiles and 20th-80th range; Rising/FallingSlope = slopes of rising / falling parts.", "cap"))

    out.parent.mkdir(exist_ok=True)
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                            bottomMargin=1.8 * cm, title="eGeMAPS feature groups", author="Aung Khant Maung (Black Box group)")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
