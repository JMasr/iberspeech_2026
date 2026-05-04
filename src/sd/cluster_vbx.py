"""VBx clustering wrapper, with a simple agglomerative-clustering fallback.

VBx (Variational Bayes HMM over x-vectors) is the recipe of choice for long
interviews; it naturally handles unknown speaker counts. The reference impl is
BUT-SpeechFIT/VBx. If it's not installed (or its native deps fail), we fall
back to scipy AHC on cosine distance — measurably worse but never broken.

Both produce a list of cluster labels parallel to the input embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VBxConfig:
    pi_init: int = 8
    fa: float = 0.3  # acoustic factor (VBx hyperparam)
    fb: float = 17.0  # speaker factor
    loop_p: float = 0.99
    max_iter: int = 40


def cluster(embeddings, *, config: VBxConfig = VBxConfig()) -> list[int]:
    """Cluster (N, D) embeddings; return one integer label per row."""
    try:
        return _vbx(embeddings, config)
    except (ImportError, RuntimeError):
        return _ahc_fallback(embeddings)


def _vbx(embeddings, config: VBxConfig) -> list[int]:  # pragma: no cover (heavy)
    import numpy as np
    from VBx.VBx import VBx as run_vbx  # type: ignore[import-not-found]

    X = np.asarray(embeddings, dtype="float32")
    init = np.zeros(X.shape[0], dtype="int32")  # warmstart from AHC if available
    init = np.asarray(_ahc_fallback(X), dtype="int32")
    labels, *_ = run_vbx(
        X,
        config.pi_init,
        config.fa,
        config.fb,
        config.loop_p,
        Q_init=init,
        maxIters=config.max_iter,
    )
    return [int(x) for x in labels]


def _ahc_fallback(embeddings) -> list[int]:
    """Cosine-distance AHC with a heuristic threshold of 0.5 on cosine distance."""
    import numpy as np
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    X = np.asarray(embeddings, dtype="float64")
    if len(X) <= 1:
        return [0] * len(X)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    Xn = X / norms
    distances = pdist(Xn, metric="cosine")
    Z = linkage(distances, method="average")
    labels = fcluster(Z, t=0.5, criterion="distance")
    return [int(x) - 1 for x in labels]
