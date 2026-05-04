"""ROVER hypothesis fusion (Recognizer Output Voting Error Reduction).

Implementation follows Fiscus 1997 with two practical adaptations:

1. **Confidence weighting** (Schwenk-style): when more than two hypotheses
   contribute to a slot, the winning word is the one with the highest summed
   confidence rather than the simple majority. We accept per-word confidences
   when present and fall back to a uniform weight of 1.0.

2. **Word-network alignment**: we build the WTN (word transition network) by
   greedy left-to-right alignment of the second-and-onwards hypotheses against
   the first using string equality. This is the canonical NIST rover behavior.

Pure Python — no torch, no transformers — so the fusion logic is unit-tested
without GPU.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WordToken:
    word: str
    confidence: float = 1.0


Hypothesis = list[WordToken]


def from_words(words: list[str], confidence: float = 1.0) -> Hypothesis:
    return [WordToken(word=w, confidence=confidence) for w in words]


def _align(reference: list[str], hypothesis: list[str]) -> list[tuple[str | None, str | None]]:
    """Return a Levenshtein alignment of two token sequences.

    Each alignment cell is a (ref_token_or_None, hyp_token_or_None) pair:
      - (a, a) when matched,
      - (a, b) when substitution (a != b),
      - (a, None) when deletion (hypothesis dropped a),
      - (None, b) when insertion (hypothesis added b).
    """
    n, m = len(reference), len(hypothesis)
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        cost[i][0] = i
    for j in range(m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            cost[i][j] = min(
                cost[i - 1][j] + 1,  # delete
                cost[i][j - 1] + 1,  # insert
                cost[i - 1][j - 1] + sub,  # match/substitute
            )

    align: list[tuple[str | None, str | None]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and reference[i - 1] == hypothesis[j - 1]:
            align.append((reference[i - 1], hypothesis[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + 1:
            align.append((reference[i - 1], hypothesis[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            align.append((reference[i - 1], None))
            i -= 1
        else:
            align.append((None, hypothesis[j - 1]))
            j -= 1
    align.reverse()
    return align


def rover(hypotheses: list[Hypothesis]) -> list[str]:
    """Fuse a list of hypotheses by ROVER voting.

    Single-hypothesis input returns its words unchanged.
    """
    if not hypotheses:
        return []
    if len(hypotheses) == 1:
        return [tok.word for tok in hypotheses[0]]

    # Use the longest hypothesis as the base — minimizes deletions during merge.
    hypotheses = sorted(hypotheses, key=lambda h: -len(h))
    base = hypotheses[0]
    base_words = [tok.word for tok in base]

    # WTN slots: each slot accumulates {word_or_NULL: total_confidence}.
    slots: list[dict[str, float]] = []
    for tok in base:
        slots.append({tok.word: tok.confidence})

    for hyp in hypotheses[1:]:
        hyp_words = [tok.word for tok in hyp]
        align = _align(base_words, hyp_words)

        # Walk the alignment; map alignment positions to slot indices.
        slot_idx = 0
        new_slots: list[dict[str, float]] = []
        h_iter = iter(hyp)
        # We may need to add insertion slots between existing ones; rebuild slots.
        for ref_tok, hyp_tok in align:
            if ref_tok is None and hyp_tok is None:
                continue
            if ref_tok is not None and hyp_tok is not None:
                # match or substitution
                conf = next(h_iter).confidence
                slot = slots[slot_idx]
                slot[hyp_tok] = slot.get(hyp_tok, 0.0) + conf
                new_slots.append(slot)
                slot_idx += 1
            elif ref_tok is not None and hyp_tok is None:
                # deletion in hyp — base has a word, hyp doesn't.
                slot = slots[slot_idx]
                slot["__NULL__"] = slot.get("__NULL__", 0.0) + 1.0
                new_slots.append(slot)
                slot_idx += 1
            else:
                # insertion in hyp — slot the hyp word as a new slot. Anchor it
                # between the previous slot and the next so downstream merges
                # remain consistent.
                conf = next(h_iter).confidence
                ins_slot = {hyp_tok: conf, "__NULL__": float(len(hypotheses) - 1)}
                new_slots.append(ins_slot)
        # Trailing base slots (if any) keep their __NULL__ counts updated.
        while slot_idx < len(slots):
            slot = slots[slot_idx]
            slot["__NULL__"] = slot.get("__NULL__", 0.0) + 1.0
            new_slots.append(slot)
            slot_idx += 1
        slots = new_slots
        base_words = [_winner(s) for s in slots]

    return [w for w in (_winner(s) for s in slots) if w != "__NULL__"]


def _winner(slot: dict[str, float]) -> str:
    return max(slot.items(), key=lambda kv: kv[1])[0]
