"""Text normalization for ALBAYZIN 2026 ASR scoring.

Two scoring tracks:
  - **raw**: punctuated, mixed case (post-edit Whisper output: truecase fix, quote/dash normalize).
  - **normalized**: lowercase, no punctuation, whitespace-collapsed.

Critical: ``normalize_for_eval`` MUST agree with the meeteval reference treatment
on the dev set or the normalized-track WER is silently inflated. Calibrate on dev
before submission (see PROGRESS.md "Evaluations needed" §8).
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import unicodedata

# Non-speech tokens organizers ban in transcriptions (eval rule). Always strip.
_NONSPEECH = re.compile(
    r"\[(?:risa|risas|susp|suspiro|tos|carraspeo|"
    r"asentimiento|negación|gemido|gruñido|"
    r"silencio|ruido|inintelig\.?|música|musica|aplausos)\]",
    re.IGNORECASE,
)

# Inline annotations like {{name}}, ((overlap)), <inaudible>. Strip wholesale.
_INLINE_ANNOT = re.compile(r"(?:\{\{[^}]*\}\}|\(\([^)]*\)\)|<[^>]*>)")

# Punctuation we strip for the normalized track. Keep apostrophe inside words
# (e.g., "pa'lante") because dialectal contractions should not be split.
_PUNCT_DROP = re.compile(r"[¿¡!?\.,:;\"“”«»—–\-—–()\[\]/]")

# Multiple whitespace → single space.
_WS = re.compile(r"\s+")

# Pattern for word-internal apostrophe handling — preserve.
# Pattern for stray leading/trailing apostrophes — strip.
_EDGE_APOS = re.compile(r"(?:^|\s)'+|'+(?=\s|$)")


def strip_nonspeech(text: str) -> str:
    """Remove organizer-banned non-speech tokens. Used by both raw and normalized paths."""
    text = _NONSPEECH.sub(" ", text)
    text = _INLINE_ANNOT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def normalize_for_eval(text: str) -> str:
    """Deterministic normalization to match meeteval reference treatment.

    Order of operations matters; do not reorder without re-calibrating on dev.
    """
    if text is None:
        return ""
    # 1. NFC unicode normalization (compose accents) — meeteval expects composed.
    text = unicodedata.normalize("NFC", text)
    # 2. Strip non-speech and inline annotations.
    text = strip_nonspeech(text)
    # 3. Lowercase.
    text = text.lower()
    # 4. Strip punctuation. Preserve apostrophe inside words.
    text = _PUNCT_DROP.sub(" ", text)
    # 5. Strip stray edge apostrophes (e.g., 'algo' → algo) but keep word-internal.
    text = _EDGE_APOS.sub(" ", text)
    # 6. Collapse whitespace.
    text = _WS.sub(" ", text).strip()
    return text


def post_edit_raw(text: str) -> str:
    """Light post-edit for the raw (punctuated) scoring track.

    - Strips banned non-speech tokens (eval rule).
    - Normalizes typographic quotes/dashes to their ASCII equivalents.
    - Collapses whitespace.

    Does NOT change case (Whisper produces appropriate casing for the dialect / register).
    """
    if text is None:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = strip_nonspeech(text)
    # Typographic quote/dash normalization — meeteval treats these inconsistently.
    text = (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("‘", "'")
        .replace("’", "'")
        .replace("—", "-")
        .replace("–", "-")
    )
    text = _WS.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Dialectal lexicon application.
# ---------------------------------------------------------------------------


class DialectLexicon:
    """Applies the curated dialect lexicon at ``data/dialect_lexicon.json``.

    Two modes:
      - ``to_canonical``: canonicalize variants → canonical (used to harmonize the
        normalized scoring track).
      - ``to_variant``: apply dialectal variant for a target province (used at
        FT data prep time when we want to match orthography).

    Both are conservative — we only apply rules whose ``applies_to`` matches the
    province (or ``"*"``).
    """

    def __init__(self, rules: list[dict]):
        self._rules = rules

    @classmethod
    def from_file(cls, path: str | Path) -> "DialectLexicon":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(rules=data.get("rules", []))

    def _matching(self, province: str | None) -> list[dict]:
        out = []
        for rule in self._rules:
            applies = rule.get("applies_to", ["*"])
            if "*" in applies or (province and province in applies):
                out.append(rule)
        return out

    def to_canonical(self, text: str, province: str | None = None) -> str:
        """Map dialect variants to canonical forms."""
        for rule in self._matching(province):
            ctx = rule.get("context", "word_whole")
            canonical = rule["canonical"]
            for variant in rule.get("variants", []):
                if variant == canonical:
                    continue
                text = self._replace(text, variant, canonical, ctx)
        return text

    def to_variant(self, text: str, province: str | None = None) -> str:
        """Apply a province's preferred variant if a single one is listed.

        This is data-prep only and intentionally conservative (uses the first variant).
        """
        for rule in self._matching(province):
            ctx = rule.get("context", "word_whole")
            canonical = rule["canonical"]
            variants = rule.get("variants", [])
            if not variants:
                continue
            text = self._replace(text, canonical, variants[0], ctx)
        return text

    @staticmethod
    def _replace(text: str, old: str, new: str, context: str) -> str:
        if context == "word_whole":
            pattern = re.compile(rf"\b{re.escape(old)}\b", re.IGNORECASE)
        elif context == "word_suffix":
            pattern = re.compile(rf"{re.escape(old)}\b", re.IGNORECASE)
        elif context == "word_prefix":
            pattern = re.compile(rf"\b{re.escape(old)}", re.IGNORECASE)
        elif context == "syllable_coda":
            # Best-effort: match the segment word-finally or before another
            # consonant. Conservative — we never apply these aggressively at
            # eval time; mostly a no-op at runtime.
            pattern = re.compile(
                rf"{re.escape(old)}(?=[bcdfghjklmnñpqrstvwxyz]|\b)", re.IGNORECASE
            )
        else:
            pattern = re.compile(re.escape(old))
        return pattern.sub(new, text)
