"""Speaker embeddings (WeSpeaker ResNet152) + PLDA backend.

We do NOT fine-tune the embedding model — the COSER speaker pool is small and
FT typically degrades performance. The PLDA backend is fit on the COSER
speaker set (small, robust).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EmbedConfig:
    backend: str = "wespeaker"  # wespeaker | titanet
    checkpoint: str | None = None
    plda_path: Path = Path("models/plda_coser.npz")
    extra: dict = field(default_factory=dict)


def load_embedder(cfg: EmbedConfig):  # pragma: no cover (heavy)
    if cfg.backend == "wespeaker":
        try:
            import wespeaker
        except ImportError as e:
            raise RuntimeError(
                "wespeaker not installed. `uv pip install -e '.[wespeaker]'`."
            ) from e
        model = wespeaker.load_model(
            "english"
        )  # public default; replace with multilingual when available
        return _WeSpeakerWrapper(model)
    if cfg.backend == "titanet":
        from nemo.collections.asr.models import EncDecSpeakerLabelModel

        model = EncDecSpeakerLabelModel.from_pretrained("titanet_large")
        return _NemoWrapper(model)
    raise ValueError(f"unknown backend {cfg.backend!r}")


class _WeSpeakerWrapper:  # pragma: no cover (heavy)
    def __init__(self, model):
        self._model = model

    def __call__(self, audio_clip, sr: int = 16_000):
        import numpy as np

        emb = self._model.extract_embedding_from_pcm(audio_clip, sr)
        return np.asarray(emb, dtype="float32")


class _NemoWrapper:  # pragma: no cover (heavy)
    def __init__(self, model):
        self._model = model

    def __call__(self, audio_clip, sr: int = 16_000):
        import torch

        with torch.no_grad():
            wav = torch.from_numpy(audio_clip).float().unsqueeze(0)
            len_t = torch.tensor([wav.shape[1]])
            _, emb = self._model.forward(input_signal=wav, input_signal_length=len_t)
        return emb.cpu().numpy()[0].astype("float32")


# ---------------------------------------------------------------------------
# PLDA backend (lightweight, numpy-only impl).
# ---------------------------------------------------------------------------


@dataclass
class PLDAModel:
    """Two-covariance PLDA. ``mu`` is the mean; ``B`` between-class; ``W`` within-class."""

    mu: object  # np.ndarray
    B: object
    W: object

    def save(self, path: str | Path) -> None:
        import numpy as np

        np.savez(str(path), mu=self.mu, B=self.B, W=self.W)

    @classmethod
    def load(cls, path: str | Path) -> "PLDAModel":
        import numpy as np

        npz = np.load(str(path))
        return cls(mu=npz["mu"], B=npz["B"], W=npz["W"])


def fit_plda(embeddings, speaker_ids) -> PLDAModel:
    """Fit a two-covariance PLDA on (N, D) embeddings with N speaker labels."""
    import numpy as np

    X = np.asarray(embeddings, dtype="float64")
    spk = np.asarray(speaker_ids)
    mu = X.mean(axis=0)
    Xc = X - mu
    classes, counts = np.unique(spk, return_counts=True)
    class_means = np.stack([Xc[spk == c].mean(axis=0) for c in classes])
    B = (class_means.T @ (class_means * counts[:, None])) / max(1, counts.sum())
    W = np.zeros_like(B)
    for c, n in zip(classes, counts):
        deviations = Xc[spk == c] - Xc[spk == c].mean(axis=0)
        W += deviations.T @ deviations
    W /= max(1, counts.sum() - len(classes))
    return PLDAModel(mu=mu, B=B, W=W)


def plda_score(model: PLDAModel, a, b) -> float:
    """Symmetric PLDA log-likelihood ratio. Higher = same speaker."""
    import numpy as np

    a = np.asarray(a, dtype="float64") - model.mu
    b = np.asarray(b, dtype="float64") - model.mu
    sigma = model.B + model.W
    inv_sigma = np.linalg.pinv(sigma)
    inv_w = np.linalg.pinv(model.W)
    inv_b = np.linalg.pinv(model.B)
    same = -0.5 * (a - b) @ inv_w @ (a - b) - 0.5 * (a + b) @ inv_b @ (a + b)
    diff = -0.5 * a @ inv_sigma @ a - 0.5 * b @ inv_sigma @ b
    return float(same - diff)
