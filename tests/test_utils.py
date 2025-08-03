import os

import numpy as np
import pytest

from audiomentations.core.utils import (
    calculate_desired_noise_rms,
    convert_decibels_to_amplitude_ratio,
    find_audio_files_in_paths,
    calculate_rms,
    calculate_rms_without_silence,
    a_weighting_frequency_envelope,
    get_crossfade_length,
)
from demo.demo import DEMO_DIR


def test_calculate_desired_noise_rms():
    noise_rms = calculate_desired_noise_rms(clean_rms=0.5, snr=6)
    assert noise_rms == pytest.approx(0.2505936168136362)


def test_convert_decibels_to_amplitude_ratio():
    amplitude_ratio = convert_decibels_to_amplitude_ratio(decibels=-6)
    assert amplitude_ratio == pytest.approx(0.5011872336272722)

    amplitude_ratio = convert_decibels_to_amplitude_ratio(decibels=6)
    assert amplitude_ratio == pytest.approx(1.9952623149688795)


def test_find_audio_files_in_paths_uppercase_extension():
    file_paths = find_audio_files_in_paths(DEMO_DIR, traverse_subdirectories=False)
    found_it = False
    for file_path in file_paths:
        if file_path.name == "stereo_24bit.WAV":
            found_it = True
            break
    assert found_it


def test_find_single_audio_file():
    file_paths = find_audio_files_in_paths(
        os.path.join(DEMO_DIR, "bus.opus"), traverse_subdirectories=False
    )
    assert len(file_paths) == 1
    assert file_paths[0].name == "bus.opus"


def test_find_multiple_audio_files():
    file_paths = find_audio_files_in_paths(
        [os.path.join(DEMO_DIR, "bus.opus"), os.path.join(DEMO_DIR, "testing.m4a")]
    )
    assert len(file_paths) == 2
    assert file_paths[0].name == "bus.opus"
    assert file_paths[1].name == "testing.m4a"


def test_find_audio_files_multiple_dirs():
    file_paths = find_audio_files_in_paths(
        [os.path.join(DEMO_DIR, "short_noises"), os.path.join(DEMO_DIR, "ir")]
    )
    assert len(file_paths) == 8
    assert file_paths[0].name == "130921_iPhone_rub_channel0_chunk83_aug2.wav"


def test_calculate_rms_stereo():
    np.random.seed(42)
    sample_len = 1024
    samples_in = np.random.uniform(low=-0.5, high=0.5, size=(2, sample_len)).astype(
        np.float32
    )
    rms = calculate_rms(samples_in)
    assert rms == pytest.approx(0.287, abs=0.01)


def test_calculate_rms_without_silence():
    sample_rate = 48000
    samples_in = np.zeros(int(2.0022 * sample_rate))
    samples_in[0:sample_rate] = 0.4 * np.ones(sample_rate)
    rms_before = calculate_rms(samples_in)
    rms_after = calculate_rms_without_silence(samples_in, sample_rate)
    assert rms_after > rms_before
    assert rms_after == pytest.approx(0.4)

    # Check that the function works if the input is shorter than a window (25 ms)
    rms_short = calculate_rms_without_silence(
        samples_in[0 : int(0.015 * sample_rate)], sample_rate
    )
    assert rms_short == pytest.approx(0.4)


@pytest.mark.parametrize(
    "sample_rate, n_fft",
    [
        (16000, 1024),
        (48000, 4096),
    ],
)
def test_a_weighting_curve(sample_rate, n_fft):
    freqs = np.fft.rfftfreq(n_fft, 1 / sample_rate)
    weighting = (
        (12194**2 * freqs**4)
        / (
            (freqs**2 + 20.6**2)
            * np.sqrt((freqs**2 + 107.7**2) * (freqs**2 + 737.9**2))
            * (freqs**2 + 12194**2)
        )
    ) + 2.00

    assert a_weighting_frequency_envelope(n_fft, sample_rate) == pytest.approx(
        weighting, abs=0.01
    )


def test_get_crossfade_duration_too_short_duration():
    with pytest.warns(UserWarning, match="crossfade_duration is too small"):
        result = get_crossfade_length(sample_rate=44_100, crossfade_duration=0.00001)
    assert result == 2


def test_get_crossfade_duration_odd_length_case():
    result = get_crossfade_length(sample_rate=10, crossfade_duration=0.3)
    # The function should add 1 to 3, resulting in an even value of 4.
    assert result == 4
