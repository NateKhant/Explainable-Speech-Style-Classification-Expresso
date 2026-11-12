"""Step 3 (own copy): leave-one-actor-out folds + a random split for comparison.

The other team will deliver the official split CSV. Until then we build the same
folds ourselves (they follow directly from speaker_id), and later check theirs
against ours with --check.

Output: splits/folds.csv with columns
  id, speaker_id, style,
  fold          1..4, the fold in which this clip is in the TEST set
                (fold k holds out speaker ex0k: ex01 -> 1, ..., ex04 -> 4)
  random_split  "train" / "test": stratified-by-style 80/20 random split (seed 42),
                reported beside the actor folds to measure speaker leakage

Usage:
    .venv/bin/python scripts/02_make_folds.py
    .venv/bin/python scripts/02_make_folds.py --check path/to/other_team_split.csv
"""

import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from config import CLIPS_CSV, FOLDS_CSV, SEED


def make_folds() -> pd.DataFrame:
    clips = pd.read_csv(CLIPS_CSV)[["id", "speaker_id", "style"]]
    assert clips["id"].is_unique
    speakers = sorted(clips["speaker_id"].unique())
    assert len(speakers) == 4, speakers
    clips["fold"] = clips["speaker_id"].map({s: i + 1 for i, s in enumerate(speakers)})

    tr, _ = train_test_split(clips.index, test_size=0.2, stratify=clips["style"], random_state=SEED)
    clips["random_split"] = "test"
    clips.loc[tr, "random_split"] = "train"
    return clips


def check(ours: pd.DataFrame, path: str) -> None:
    theirs = pd.read_csv(path)
    print(f"Their file: {theirs.shape}, columns {list(theirs.columns)}")
    a, b = set(ours["id"]), set(theirs["id"])
    print(f"ids only in ours: {len(a - b)}, only in theirs: {len(b - a)}, shared: {len(a & b)}")
    m = ours.merge(theirs, on="id", suffixes=("", "_theirs"))
    for col in ["speaker_id", "style"]:
        if f"{col}_theirs" in m:
            print(f"{col} mismatches: {(m[col] != m[f'{col}_theirs']).sum()}")
    # their fold label may be a number or a speaker name: compare the grouping, not the label
    fold_col = next((c for c in ["fold_theirs", "fold", "test_speaker"] if c in m and c != "fold"), None)
    if fold_col:
        same = m.groupby(fold_col)["fold"].nunique().eq(1).all() and m.groupby("fold")[fold_col].nunique().eq(1).all()
        print(f"fold grouping identical ({fold_col}): {same}")
        print(pd.crosstab(m["fold"], m[fold_col]).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", help="other team's split CSV to compare against ours")
    args = ap.parse_args()

    folds = make_folds()
    if args.check:
        check(folds, args.check)
        return

    FOLDS_CSV.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(FOLDS_CSV, index=False)
    print(f"Saved {folds.shape} -> {FOLDS_CSV}\n")
    for k, g in folds.groupby("fold"):
        print(f"fold {k}: test {g['speaker_id'].iloc[0]} n={len(g)}, train n={len(folds) - len(g)}")
    print("\nTest clips per style x fold:")
    print(pd.crosstab(folds["style"], folds["fold"]).to_string())
    print("\nRandom split:")
    print(pd.crosstab(folds["style"], folds["random_split"]).to_string())


if __name__ == "__main__":
    main()
