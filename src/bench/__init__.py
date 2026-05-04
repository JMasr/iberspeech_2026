"""Public-data benchmarks for pipeline validation.

WHAT THIS IS
    A separate, sandboxed harness that runs the COSER pipeline on PUBLIC datasets
    (Common Voice ES, VoxConverse) so we can verify the code paths and the main
    hypotheses (FT lift, fusion lift, boundary-snap lift, non-speech mask gain)
    BEFORE the COSER data drops.

WHAT THIS IS NOT
    This module MUST NOT feed any audio into COSER fine-tuning. The ALBAYZIN
    Speech-COSER eval rules forbid external audio for training. Bench data is
    validation-only — never reused as training input. Each bench writes to
    ``data/bench/<dataset>/`` and the main FT recipes (`src/asr/whisper_ft.py`,
    `src/asr/w2v2_ft.py`) refuse manifests that originate from this directory.
"""

BENCH_DATA_ROOT = "data/bench"
BENCH_DISALLOWED_FOR_FT = (BENCH_DATA_ROOT,)

# Tag values that the FT recipes refuse. Any manifest row whose ``source``
# matches one of these is dropped before training so external bench audio
# cannot leak into the COSER FT (eval-rule compliance).
BENCH_SOURCE_TAGS = frozenset(
    {"cv_es", "cv_multilingual", "voxconverse", "synthetic", "synth_coser"}
)


def assert_no_bench_rows(manifest_df) -> None:
    """Raise if a training manifest contains bench-tagged rows.

    Called by ``src.asr.whisper_ft`` and ``src.asr.w2v2_ft`` before training.
    """
    if "source" not in manifest_df.columns:
        return
    bad = manifest_df[manifest_df["source"].isin(BENCH_SOURCE_TAGS)]
    if len(bad):
        raise ValueError(
            f"Training manifest contains {len(bad)} rows tagged as bench data "
            f"({sorted(bad['source'].unique())}). External audio is forbidden "
            "by the ALBAYZIN Speech-COSER eval rules. Drop these rows or use "
            "data/raw/ inputs only."
        )
