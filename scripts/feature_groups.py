"""Grouping of the 88 eGeMAPSv02 functionals into 7 families of correlated features.

Used for actionable recourse (proposal Section 6.4): a narrator can change speaking
rate, pauses and loudness on the next take, but not their timbre, pitch level
(a proxy for speaker sex) or vocal-tract shape. Decision 2026-11-12.
"""

# (group name, changeable by the narrator?, rule matching a feature name)
GROUPS = [
    ("Rate & pauses", "Yes", lambda f: f in {
        "loudnessPeaksPerSec", "VoicedSegmentsPerSec", "MeanVoicedSegmentLengthSec",
        "StddevVoicedSegmentLengthSec", "MeanUnvoicedSegmentLength", "StddevUnvoicedSegmentLength"}),
    ("Loudness", "Yes", lambda f: f.startswith("loudness_sma3") or f == "equivalentSoundLevel_dBp"),
    ("Pitch movement", "Partly", lambda f: f.startswith("F0semitone") and not f.endswith(
        ("_amean", "percentile20.0", "percentile50.0", "percentile80.0"))),
    ("Pitch level", "No (sex proxy)", lambda f: f.startswith("F0semitone") and f.endswith(
        ("_amean", "percentile20.0", "percentile50.0", "percentile80.0"))),
    ("Voice quality", "Partly", lambda f: f.startswith(("jitter", "shimmer", "HNR", "logRelF0-H1"))),
    ("Spectral / timbre", "No", lambda f: f.startswith(("alphaRatio", "hammarberg", "slope", "spectralFlux", "mfcc"))),
    ("Formants", "No", lambda f: f[:2] in {"F1", "F2", "F3"}),
]
DESCRIPTION = {
    "Rate & pauses": "speaking rate (loudness peaks / s, voiced segments / s), voiced and pause segment lengths",
    "Loudness": "loudness mean, spread, percentiles, rise / fall slopes; equivalent sound level",
    "Pitch movement": "F0 variation and range, rising / falling pitch slopes (intonation)",
    "Pitch level": "F0 mean and 20th / 50th / 80th percentiles (a proxy for speaker sex)",
    "Voice quality": "jitter, shimmer, harmonics-to-noise ratio, H1-H2, H1-A3 (breathiness, roughness)",
    "Spectral / timbre": "alpha ratio, Hammarberg index, spectral slopes and flux, MFCC 1-4",
    "Formants": "F1-F3 frequency, bandwidth and amplitude (vocal-tract shape, articulation)",
}
GROUP_NAMES = [g for g, _, _ in GROUPS]
CHANGEABLE = {g: c for g, c, _ in GROUPS}


def group_of(feature: str) -> str:
    hits = [g for g, _, rule in GROUPS if rule(feature)]
    if len(hits) != 1:
        raise ValueError(f"{feature} matches {hits}; every feature must be in exactly one group")
    return hits[0]


def feature_groups(features) -> dict:
    """feature name -> group name; checks that all features are covered exactly once."""
    mapping = {f: group_of(f) for f in features}
    return mapping
