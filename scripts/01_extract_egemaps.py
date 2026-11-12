"""Step 2 (baseline path): extract the 88 eGeMAPSv02 functionals for every clip.

Per clip:
  - read the original 48 kHz audio from the Hugging Face cache (soundfile, no ffmpeg)
  - mix to mono if needed, resample 48 kHz -> 16 kHz (polyphase, scipy)
  - full clip: no cropping or padding (padding with silence would distort
    pause / loudness statistics)
  - no loudness normalisation (loudness stays a usable, changeable feature)
  - openSMILE eGeMAPSv02, functionals level -> 88 numbers

Output: features/egemaps.parquet with columns id, speaker_id, style + 88 features.

Usage:
    .venv/bin/python scripts/01_extract_egemaps.py [--workers N] [--limit N]
"""

import argparse
import io
import os
import time
from math import gcd
from multiprocessing import Pool

import numpy as np
import opensmile
import pandas as pd
import soundfile as sf
from datasets import Audio, load_dataset
from scipy.signal import resample_poly

from config import CACHE, CONFIG, DATASET_ID, FEATURES, SPLIT, STYLES, TARGET_SR

_smile = None


def _init_worker() -> None:
    global _smile
    _smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )


def load_16k(audio_bytes: bytes) -> np.ndarray:
    x, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    x = x.mean(axis=1)  # mono
    if sr != TARGET_SR:
        g = gcd(sr, TARGET_SR)
        x = resample_poly(x, TARGET_SR // g, sr // g).astype("float32")
    return x


def extract(item):
    clip_id, speaker_id, style, audio_bytes = item
    x = load_16k(audio_bytes)
    feats = _smile.process_signal(x, TARGET_SR).iloc[0]
    return clip_id, speaker_id, style, len(x) / TARGET_SR, feats.to_numpy(dtype="float64")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=None, help="only the first N clips (testing)")
    args = ap.parse_args()

    ds = load_dataset(DATASET_ID, CONFIG, split=SPLIT, cache_dir=str(CACHE))
    ds = ds.cast_column("audio", Audio(decode=False))
    styles = ds["style"]
    keep = [i for i, s in enumerate(styles) if s in STYLES]
    if args.limit:
        keep = keep[: args.limit]
    ds = ds.select(keep)
    print(f"{len(ds)} clips, {args.workers} workers")

    def items():
        for row in ds:
            yield row["id"], row["speaker_id"], row["style"], row["audio"]["bytes"]

    _init_worker()
    names = list(_smile.feature_names)
    assert len(names) == 88, len(names)

    rows, t0 = [], time.time()
    with Pool(args.workers, initializer=_init_worker) as pool:
        for n, r in enumerate(pool.imap(extract, items(), chunksize=16), 1):
            rows.append(r)
            if n % 1000 == 0 or n == len(ds):
                print(f"  {n}/{len(ds)}  ({time.time() - t0:.0f}s)", flush=True)

    meta = pd.DataFrame([r[:4] for r in rows], columns=["id", "speaker_id", "style", "duration_s"])
    feats = pd.DataFrame(np.vstack([r[4] for r in rows]), columns=names)
    df = pd.concat([meta, feats], axis=1)

    assert df["id"].is_unique, "duplicate clip ids"
    n_nan = int(feats.isna().sum().sum())
    print(f"NaN values: {n_nan}; all-constant features: {list(feats.columns[feats.nunique() <= 1])}")

    out = FEATURES if not args.limit else FEATURES.with_name(f"egemaps_first{args.limit}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Saved {df.shape} -> {out}")


if __name__ == "__main__":
    main()
