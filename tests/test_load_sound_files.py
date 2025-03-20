import math
import os
import pytest
import warnings

import numpy as np

from audiomentations.core.audio_loading_utils import (
    load_sound_file,
)
from demo.demo import DEMO_DIR


def test_load_stereo_ogg_vorbis():
    samples, sample_rate = load_sound_file(
        os.path.join(DEMO_DIR, "background_noises", "hens.ogg"), sample_rate=None
    )
    assert samples.dtype == np.float32
    assert samples.ndim == 1

    # Apparently, the exact duration may vary slightly based on which decoder is used
    assert samples.shape[0] >= 442575
    assert samples.shape[0] <= 443328

    max_value = np.amax(samples)
    assert max_value > 0.02
    assert max_value < 1.0

def test_load_mono_opus():
    samples, sample_rate = load_sound_file(
        os.path.join(DEMO_DIR, "bus.opus"), sample_rate=None
    )
    assert samples.dtype == np.float32
    assert samples.ndim == 1

    # Apparently, the exact duration may vary slightly based on which decoder is used
    assert samples.shape[0] >= 36682
    assert samples.shape[0] <= 36994

    max_value = np.amax(samples)
    assert max_value > 0.3
    assert max_value < 1.0

def test_load_mono_m4a():
    samples, sample_rate = load_sound_file(
        os.path.join(DEMO_DIR, "testing.m4a"), sample_rate=None
    )
    assert sample_rate == 44100
    assert samples.dtype == np.float32
    assert samples.ndim == 1

    max_value = np.amax(samples)
    assert max_value > 0.1
    assert max_value < 1.0

def test_load_mono_signed_16_bit_wav():
    samples, sample_rate = load_sound_file(
        os.path.join(DEMO_DIR, "acoustic_guitar_0.wav"), sample_rate=None
    )
    assert sample_rate == 16000
    assert samples.dtype == np.float32
    assert samples.ndim == 1

    assert samples.shape[0] == 140544

    max_value = np.amax(samples)
    assert max_value > 0.5
    assert max_value < 1.0

def test_load_stereo_signed_16_bit_wav():
    samples, sample_rate = load_sound_file(
        os.path.join(DEMO_DIR, "stereo_16bit.wav"), sample_rate=None
    )
    assert sample_rate == 16000
    assert samples.dtype == np.float32
    assert samples.ndim == 1

    assert samples.shape[0] == 17833

    max_value = np.amax(samples)
    assert max_value > 0.5
    assert max_value < 1.0

def test_load_mono_signed_24_bit_wav():
    samples, sample_rate = load_sound_file(
        os.path.join(DEMO_DIR, "signed_24bit.wav"), sample_rate=None
    )
    assert sample_rate == 48000
    assert samples.dtype == np.float32
    assert samples.ndim == 1

    assert samples.shape[0] == 54514

    max_value = np.amax(samples)
    assert max_value > 0.09
    assert max_value < 1.0

def test_load_mono_signed_24_bit_wav2():
    samples, sample_rate = load_sound_file(
        os.path.join(DEMO_DIR, "mono_int24.wav"), sample_rate=None
    )
    assert sample_rate == 44100
    assert samples.dtype == np.float32
    assert samples.ndim == 1

    assert samples.shape[-1] == 22

    max_value = np.amax(samples)
    assert max_value == pytest.approx(0.96750367)
    min_value = np.amin(samples)
    assert min_value == pytest.approx(-0.9822748)

def test_load_mono_signed_32_bit_wav():
    samples, sample_rate = load_sound_file(
        os.path.join(DEMO_DIR, "mono_int32.wav"), sample_rate=None
    )
    assert sample_rate == 44100
    assert samples.dtype == np.float32
    assert samples.ndim == 1

    assert samples.shape[-1] == 22

    max_value = np.amax(samples)
    assert max_value == pytest.approx(0.96750367)
    min_value = np.amin(samples)
    assert min_value == pytest.approx(-0.9822748)

def test_load_mono_float64_wav():
    samples, sample_rate = load_sound_file(
        os.path.join(DEMO_DIR, "mono_float64.wav"), sample_rate=None
    )
    assert sample_rate == 44100
    assert samples.dtype == np.float32
    assert samples.ndim == 1

    assert samples.shape[-1] == 22

    max_value = np.amax(samples)
    assert max_value == pytest.approx(0.96750367)
    min_value = np.amin(samples)
    assert min_value == pytest.approx(-0.9822748)

def test_load_stereo_signed_24_bit_wav():
    samples, sample_rate = load_sound_file(
        os.path.join(DEMO_DIR, "stereo_24bit.WAV"), sample_rate=None
    )
    assert sample_rate == 16000
    assert samples.dtype == np.float32
    assert samples.ndim == 1

    assert samples.shape[0] == 17833

    max_value = np.amax(samples)
    assert max_value > 0.5
    assert max_value < 1.0

def test_load_mono_ms_adpcm():
    samples, sample_rate = load_sound_file(
        os.path.join(DEMO_DIR, "ms_adpcm.wav"), sample_rate=None
    )
    assert sample_rate == 11024
    assert samples.dtype == np.float32
    assert samples.ndim == 1

    assert samples.shape[0] == 895500

    max_value = np.amax(samples)
    assert max_value > 0.3
    assert max_value < 1.0

def test_load_mono_ms_adpcm_and_resample():
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")

        samples, sample_rate = load_sound_file(
            os.path.join(DEMO_DIR, "ms_adpcm.wav"), sample_rate=16000
        )

        assert len(w) >= 1

    assert sample_rate == 16000
    assert samples.dtype == np.float32
    assert samples.ndim == 1

    assert samples.shape[0] == math.ceil(895500 * 16000 / 11024)

    max_value = np.amax(samples)
    assert max_value > 0.3
    assert max_value < 1.0
