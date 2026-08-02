"""Coverage for the surviving Tier-2 surface: toggles and diagnostics.

blip25-mbe 0.3.0 is a pure bits-in/bits-out codec with no runtime
levers. The only two remaining Vocoder toggles are tone detection and
the post-decode enhancement mode; everything else (the 0.2.x encoder
knobs, denoiser front-ends, and synthesis-variant selector) was removed
upstream.
"""

import numpy as np

import blip25_mbe


def test_defaults_match_library_defaults() -> None:
    vc = blip25_mbe.Vocoder(blip25_mbe.Rate.IMBE_7200X4400)
    # 0.3.0 `Vocoder::new` defaults enhancement to NONE — the shipped
    # decode path is the reference console codec unaltered.
    assert vc.enhancement == blip25_mbe.EnhancementMode.NONE
    assert vc.tone_detection is False


def test_toggles_round_trip() -> None:
    vc = blip25_mbe.Vocoder(blip25_mbe.Rate.AMBEPLUS2_3600X2450)

    vc.set_enhancement(blip25_mbe.EnhancementMode.CLASSICAL)
    assert vc.enhancement == blip25_mbe.EnhancementMode.CLASSICAL
    vc.set_enhancement(blip25_mbe.EnhancementMode.NONE)
    assert vc.enhancement == blip25_mbe.EnhancementMode.NONE

    vc.set_tone_detection(True)
    assert vc.tone_detection is True
    vc.set_tone_detection(False)
    assert vc.tone_detection is False


def test_last_output_kind_after_encode() -> None:
    vc = blip25_mbe.Vocoder(blip25_mbe.Rate.IMBE_7200X4400)
    # No encode yet → diagnostic is None.
    assert vc.last_output_kind() is None

    silent = np.zeros(vc.frame_samples, dtype=np.int16)
    vc.encode_pcm(silent)
    kind = vc.last_output_kind()
    assert kind in ("voice", "silence", "tone")


def test_tone_detection_toggle_and_diagnostic_are_wellformed() -> None:
    # `set_tone_detection` is a persistent toggle. In 0.3.0 the Annex-T
    # tone overlay is applied only by the whole-signal batch encoder, not
    # by the streaming single-frame `encode_pcm` path exposed here, so a
    # clean tone through `encode_pcm` classifies as voice and
    # `last_tone_detection` stays None. The diagnostic must remain
    # well-formed (never raise) regardless.
    vc = blip25_mbe.Vocoder(blip25_mbe.Rate.AMBEPLUS2_3600X2450)
    vc.set_tone_detection(True)
    assert vc.tone_detection is True

    n = vc.frame_samples
    t = np.arange(n) / 8000.0
    pcm = (10000 * np.sin(2 * np.pi * 1031.25 * t)).astype(np.int16)

    vc.encode_pcm(pcm)
    detection = vc.last_tone_detection()
    assert detection is None or (
        0 <= detection[0] <= 255 and 0 <= detection[1] <= 127
    )
    # The toggle persists across reset.
    vc.reset()
    assert vc.tone_detection is True
