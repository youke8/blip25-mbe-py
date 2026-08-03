"""Erasure injection and concealment reporting.

Covers the packet-transport story: when a frame is lost, the caller feeds
`rate.erasure_frame()` in its place and reads back what the decoder did with
it. Skipping the frame instead would silently desynchronize decoder state
from the sender's.
"""

import numpy as np
import pytest

import blip25_mbe


ALL_RATES = [
    blip25_mbe.Rate.IMBE_7200X4400,
    blip25_mbe.Rate.IMBE_4400X4400,
    blip25_mbe.Rate.AMBEPLUS2_3600X2450,
    blip25_mbe.Rate.AMBEPLUS2_2450X2450,
]

# The mute gate emits uniform noise on [-5, 5].
COMFORT_NOISE_PEAK = 5


def _speech(n_frames: int, frame_samples: int) -> np.ndarray:
    t = np.arange(n_frames * frame_samples)
    return (np.sin(t * 0.3) * 6000).astype(np.int16)


@pytest.mark.parametrize("rate", ALL_RATES)
def test_erasure_frame_is_one_wire_frame(rate: blip25_mbe.Rate) -> None:
    frame = rate.erasure_frame()
    assert isinstance(frame, bytes)
    assert len(frame) == rate.fec_frame_bytes


@pytest.mark.parametrize("rate", ALL_RATES)
def test_vocoder_erasure_frame_matches_rate(rate: blip25_mbe.Rate) -> None:
    vc = blip25_mbe.Vocoder(rate)
    assert vc.erasure_frame == rate.erasure_frame()


@pytest.mark.parametrize("rate", ALL_RATES)
def test_no_decode_yet_reports_nothing(rate: blip25_mbe.Rate) -> None:
    vc = blip25_mbe.Vocoder(rate)
    assert vc.last_disposition() is None
    assert vc.last_decode_errors() is None


@pytest.mark.parametrize("rate", ALL_RATES)
def test_clean_frame_reports_use(rate: blip25_mbe.Rate) -> None:
    vc = blip25_mbe.Vocoder(rate)
    enc = blip25_mbe.Vocoder(rate)
    pcm = _speech(4, vc.frame_samples)
    # The encoder carries one frame of algorithmic delay, so its first output
    # is a priming frame rather than coded speech.
    for i in range(4):
        chunk = pcm[i * vc.frame_samples : (i + 1) * vc.frame_samples]
        vc.decode_bits(enc.encode_pcm(chunk))
        assert vc.last_disposition() == "use"
        if i > 0:
            assert vc.last_decode_errors() == (0, 0)


@pytest.mark.parametrize("rate", ALL_RATES)
def test_erasures_repeat_then_mute(rate: blip25_mbe.Rate) -> None:
    """Three erasures repeat the last good frame; the fourth mutes."""
    vc = blip25_mbe.Vocoder(rate)
    enc = blip25_mbe.Vocoder(rate)
    pcm = _speech(1, vc.frame_samples)
    vc.decode_bits(enc.encode_pcm(pcm))
    assert vc.last_disposition() == "use"

    seen = []
    for _ in range(5):
        vc.decode_bits(vc.erasure_frame)
        seen.append(vc.last_disposition())

    assert seen == ["repeat", "repeat", "repeat", "mute", "mute"]


@pytest.mark.parametrize("rate", ALL_RATES)
def test_mute_output_is_comfort_noise(rate: blip25_mbe.Rate) -> None:
    vc = blip25_mbe.Vocoder(rate)
    out = None
    for _ in range(6):
        out = vc.decode_bits(vc.erasure_frame)
    assert vc.last_disposition() == "mute"
    assert int(np.abs(out.astype(np.int32)).max()) <= COMFORT_NOISE_PEAK


@pytest.mark.parametrize("rate", ALL_RATES)
def test_decoder_recovers_after_an_erasure_run(rate: blip25_mbe.Rate) -> None:
    """A good frame after a concealed run is decoded normally again."""
    vc = blip25_mbe.Vocoder(rate)
    enc = blip25_mbe.Vocoder(rate)
    frame_bits = [
        enc.encode_pcm(_speech(1, vc.frame_samples)[: vc.frame_samples])
        for _ in range(2)
    ]
    vc.decode_bits(frame_bits[0])
    for _ in range(5):
        vc.decode_bits(vc.erasure_frame)
    assert vc.last_disposition() == "mute"

    vc.decode_bits(frame_bits[1])
    assert vc.last_disposition() == "use"


@pytest.mark.parametrize("rate", ALL_RATES)
def test_reset_clears_concealment_reporting(rate: blip25_mbe.Rate) -> None:
    vc = blip25_mbe.Vocoder(rate)
    vc.decode_bits(vc.erasure_frame)
    assert vc.last_disposition() == "repeat"
    vc.reset()
    assert vc.last_disposition() is None
    assert vc.last_decode_errors() is None
