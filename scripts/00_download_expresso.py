"""Download the Expresso read set (ylacombe/expresso, config "read") from Hugging Face.

- Downloads ~5.76 GB of parquet files into ./.hf_cache (no second copy is saved).
- Prints clip counts per style and per speaker (before and after filtering).
- Writes clips.csv: id, speaker_id, style, text, sample_rate, duration_s
  for the 7 project styles (singing and anything else dropped).
  Join it with the other team's split CSV on `id`.

Audio is read with soundfile (decode=False), so ffmpeg/torch are not needed.

Usage:
    .venv/bin/python scripts/00_download_expresso.py

Later, load the audio with:
    from datasets import load_dataset, Audio
    ds = load_dataset("ylacombe/expresso", "read", split="train", cache_dir=".hf_cache")
    ds = ds.cast_column("audio", Audio(decode=False))   # then soundfile.read(io.BytesIO(row["audio"]["bytes"]))
"""

import io

import pandas as pd
import soundfile as sf
from datasets import Audio, load_dataset

from config import CACHE, CLIPS_CSV as OUT_CSV, CONFIG, DATASET_ID, SPLIT, STYLES


def main() -> None:
    print(f"Downloading {DATASET_ID} ({CONFIG}/{SPLIT}) into {CACHE} ...")
    ds = load_dataset(DATASET_ID, CONFIG, split=SPLIT, cache_dir=str(CACHE))
    ds = ds.cast_column("audio", Audio(decode=False))
    print(ds)

    meta = ds.select_columns(["id", "speaker_id", "style", "text"]).to_pandas()
    print("\nAll styles (before filtering):")
    print(meta["style"].value_counts().sort_index().to_string())
    print("\nAll speakers (before filtering):")
    print(meta["speaker_id"].value_counts().sort_index().to_string())

    unexpected = sorted(set(meta["style"]) - set(STYLES))
    print(f"\nDropping styles: {unexpected}")

    keep = meta.index[meta["style"].isin(STYLES)].tolist()
    sub = ds.select(keep)

    rates, durations = [], []
    for row in sub.select_columns(["audio"]):
        info = sf.info(io.BytesIO(row["audio"]["bytes"]))
        rates.append(info.samplerate)
        durations.append(info.frames / info.samplerate)

    clips = meta.loc[keep].reset_index(drop=True)
    clips["sample_rate"] = rates
    clips["duration_s"] = [round(d, 3) for d in durations]
    clips.to_csv(OUT_CSV, index=False)

    print(f"\nKept {len(clips)} clips -> {OUT_CSV}")
    print("\nClips per style x speaker:")
    print(pd.crosstab(clips["style"], clips["speaker_id"], margins=True).to_string())
    print("\nDuration (s) per style:")
    print(clips.groupby("style")["duration_s"].describe()[["mean", "min", "max"]].round(2).to_string())
    print("\nSample rates:", clips["sample_rate"].value_counts().to_dict())
    print(f"Total hours: {clips['duration_s'].sum() / 3600:.2f}")


if __name__ == "__main__":
    main()
