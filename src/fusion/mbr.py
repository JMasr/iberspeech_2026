"""Minimum Bayes Risk decoding over Whisper n-best.

Risk = expected WER under the posterior implied by the n-best scores. We
compute, for each candidate ``c``, ``risk(c) = sum_i p_i * WER(c, h_i)``
and return the argmin. Posterior is softmax over the n-best avg-logprobs (a
temperature of 1.0 is fine when the scores are already on a comparable scale).

WER is computed with the standard Levenshtein over whitespace tokens.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NBestEntry:
    text: str
    score: float  # avg-logprob (the higher, the more likely)


def wer(reference: list[str], hypothesis: list[str]) -> float:
    """Word error rate, normalized by reference length. Empty reference → 1.0 if hyp non-empty else 0."""
    n, m = len(reference), len(hypothesis)
    if n == 0:
        return 0.0 if m == 0 else 1.0
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        cost[i][0] = i
    for j in range(m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            cost[i][j] = min(
                cost[i - 1][j] + 1,
                cost[i][j - 1] + 1,
                cost[i - 1][j - 1] + sub,
            )
    return cost[n][m] / float(n)


def mbr(nbest: list[NBestEntry]) -> NBestEntry:
    """Pick the candidate that minimizes expected WER under the n-best posterior."""
    if not nbest:
        raise ValueError("MBR requires a non-empty n-best list")
    if len(nbest) == 1:
        return nbest[0]

    # Softmax over scores. Avg-logprobs are already in log-space.
    import math

    max_s = max(e.score for e in nbest)
    exp_scores = [math.exp(e.score - max_s) for e in nbest]
    z = sum(exp_scores)
    posterior = [s / z for s in exp_scores]

    tokens = [e.text.split() for e in nbest]
    risks = []
    for i, c in enumerate(tokens):
        risk = 0.0
        for j, h in enumerate(tokens):
            if i == j:
                continue
            risk += posterior[j] * wer(c, h)
        risks.append(risk)
    best_i = min(range(len(risks)), key=lambda i: risks[i])
    return nbest[best_i]
