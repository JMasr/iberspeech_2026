"""N-best generation and CTC lattice export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NBestRow:
    chunk_id: str
    rank: int
    text: str
    avg_logprob: float
    no_speech_prob: float = 0.0


def whisper_nbest(
    audio,
    *,
    model,
    processor,
    n_best: int = 5,
    language: str = "es",
    logits_processor=None,
) -> list[NBestRow]:  # pragma: no cover (heavy)
    """Run Whisper with beam search and return n-best.

    ``model`` is a ``transformers.WhisperForConditionalGeneration``;
    ``processor`` is a ``WhisperProcessor``. We force ``language=es`` by default
    but the caller may pass None to allow auto-LID for known multilingual
    provinces (e.g. parts of Galicia/Catalonia).
    """
    import torch

    inputs = processor(audio, sampling_rate=16_000, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    forced_decoder_ids = None
    if language is not None:
        forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            num_beams=max(n_best, 5),
            num_return_sequences=n_best,
            return_dict_in_generate=True,
            output_scores=True,
            no_repeat_ngram_size=3,
            forced_decoder_ids=forced_decoder_ids,
            logits_processor=logits_processor,
        )
    sequences = out.sequences
    scores = out.sequences_scores if hasattr(out, "sequences_scores") else None
    rows: list[NBestRow] = []
    for rank, seq in enumerate(sequences):
        text = processor.batch_decode(seq.unsqueeze(0), skip_special_tokens=True)[0]
        rows.append(
            NBestRow(
                chunk_id="",
                rank=rank,
                text=text.strip(),
                avg_logprob=float(scores[rank]) if scores is not None else 0.0,
            )
        )
    return rows


def export_ctc_lattice(  # pragma: no cover (heavy)
    logits, vocab: list[str], *, beam_width: int = 50
):
    """Export a token-level CTC lattice from XLS-R logits.

    Implementation defers to ``pyctcdecode``; we expose the function so callers
    can persist confusion networks to disk for ROVER consumption.
    """
    from pyctcdecode import build_ctcdecoder

    decoder = build_ctcdecoder(vocab)
    return decoder.decode_beams(logits.cpu().numpy(), beam_width=beam_width)


def save_nbest(rows: list[NBestRow], path: str | Path) -> Path:
    import pandas as pd

    df = pd.DataFrame([row.__dict__ for row in rows])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path
