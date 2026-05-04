"""Stage 0B — BEATs AudioSet tagger applied to chunks flagged in Stage 0A.

Default: BEATs (Microsoft, 2022) — `microsoft/BEATs-AS2M`.
Fallback: PANNs CNN14 (`Cnn14_mAP=0.431.pth`) if BEATs install/inference fails.

Both models output AudioSet-style multi-label tags; we keep only a small
allowlist of classes whose downstream consumer is concrete:

  - `Music`, `Singing`            → confirm music routing (Stage 0 → ASR skip).
  - `Laughter`, `Breath`, `Cough`,
    `Applause`, `Animal`, `Vehicle`
                                  → ASR token-drop mask (eval rule compliance).

The output is written by *updating* the existing parquet from Stage 0A: we
overwrite the `beats_tags` and `beats_scores` columns for flagged rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# AudioSet class names we keep. Lowercase for matching across PANNs/BEATs vocabs.
ALLOWED_TAGS = {
    "music",
    "singing",
    "laughter",
    "breath",
    "breathing",
    "cough",
    "applause",
    "animal",
    "domestic_animal",
    "vehicle",
    "car",
    "engine",
}

# Tags that drive the ASR token-drop mask.
NONSPEECH_TAGS = {
    "laughter",
    "breath",
    "breathing",
    "cough",
    "applause",
    "animal",
    "domestic_animal",
    "vehicle",
    "car",
    "engine",
}

# Tags that confirm music regions.
MUSIC_TAGS = {"music", "singing"}


@dataclass(frozen=True)
class BeatsConfig:
    backend: str = "beats"  # "beats" or "panns"
    checkpoint: str | None = None
    score_threshold: float = 0.30
    chunk_s: float = 30.0


def tag_chunks(
    parquet_path: str | Path,
    audio_path: str | Path,
    *,
    config: BeatsConfig = BeatsConfig(),
) -> Path:
    """Run BEATs (or PANNs fallback) over flagged chunks and update the parquet.

    Heavy imports are local. Returns the parquet path it modified.
    """
    import pandas as pd

    parquet_path = Path(parquet_path)
    df = pd.read_parquet(parquet_path)
    if "flagged_for_beats" not in df.columns:
        raise KeyError("parquet missing 'flagged_for_beats' — did you skip Stage 0A?")
    flagged = df[df["flagged_for_beats"]].copy()
    if flagged.empty:
        return parquet_path

    if config.backend == "beats":
        tagger = _load_beats(config.checkpoint)
    elif config.backend == "panns":
        tagger = _load_panns(config.checkpoint)
    else:
        raise ValueError(f"unknown backend {config.backend!r}")

    from src.data.audio import load_mono_16k, slice_audio

    audio = load_mono_16k(audio_path)
    out_tags: list[list[str] | None] = [None] * len(df)
    out_scores: list[list[float] | None] = [None] * len(df)
    for idx in flagged.index:
        clip = slice_audio(audio, float(df.at[idx, "start_s"]), float(df.at[idx, "end_s"]))
        labels, scores = tagger(clip)
        kept = [
            (lbl, sc)
            for lbl, sc in zip(labels, scores)
            if sc >= config.score_threshold and lbl.lower() in ALLOWED_TAGS
        ]
        out_tags[idx] = [lbl.lower() for lbl, _ in kept]
        out_scores[idx] = [float(sc) for _, sc in kept]

    df["beats_tags"] = out_tags
    df["beats_scores"] = out_scores
    df.to_parquet(parquet_path, index=False)
    return parquet_path


def _load_beats(checkpoint: str | None):
    """Load the vendored BEATs checkpoint and return a callable (audio→labels,scores).

    The BEATs upstream (microsoft/unilm) does not publish a pip wheel; the user
    is expected to clone the inference loader and place the checkpoint at
    ``models/beats/BEATs_iter3_plus_AS2M.pt``. We attempt the import lazily and
    raise a helpful error if absent.
    """
    try:
        from src.asr import _beats_loader  # local vendored module, optional
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "BEATs not installed. Vendor the inference loader from microsoft/unilm/beats "
            "or use config.backend='panns' as a fallback."
        ) from e
    return _beats_loader.build(checkpoint or "models/beats/BEATs_iter3_plus_AS2M.pt")


def _load_panns(checkpoint: str | None):  # pragma: no cover (heavy)
    """PANNs CNN14 fallback. Same callable signature as BEATs."""
    import numpy as np
    import torch
    from torchaudio.models import HUBERTBaseModel  # noqa: F401  (placeholder import)

    try:
        from panns_inference import AudioTagging
    except ImportError as e:
        raise RuntimeError(
            "panns_inference not installed. `pip install panns-inference` or use BEATs."
        ) from e
    tagger = AudioTagging(
        checkpoint_path=checkpoint, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    labels = tagger.labels  # type: ignore[attr-defined]

    def _call(audio):
        clipwise_output, _ = tagger.inference(audio[None, :])
        scores = clipwise_output[0]
        idx = np.argsort(-scores)[:50]
        return [labels[i] for i in idx], [float(scores[i]) for i in idx]

    return _call


def nonspeech_intervals(parquet_path: str | Path) -> list[tuple[float, float, str]]:
    """Reduce per-chunk BEATs tags to flat (start_s, end_s, tag) intervals.

    Used by ``src.asr.nonspeech_mask`` to drop tokens at inference time. We keep
    only ``NONSPEECH_TAGS`` — music is handled separately at chunk routing time.
    """
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    out: list[tuple[float, float, str]] = []
    for _, row in df.iterrows():
        tags = row.get("beats_tags")
        if not tags:
            continue
        for tag in tags:
            if tag in NONSPEECH_TAGS:
                out.append((float(row["start_s"]), float(row["end_s"]), str(tag)))
    return out


def music_intervals(parquet_path: str | Path) -> list[tuple[float, float]]:
    """Return regions that should be skipped by ASR (music/singing confirmed)."""
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    intervals: list[tuple[float, float]] = []
    for _, row in df.iterrows():
        ina_label = str(row.get("ina_label", ""))
        beats_tags = row.get("beats_tags") or []
        if ina_label.lower() in {"music", "noenergy"} or any(t in MUSIC_TAGS for t in beats_tags):
            intervals.append((float(row["start_s"]), float(row["end_s"])))
    return intervals
