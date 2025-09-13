import os
import random
import warnings

import librosa
import numpy as np
import pytest

from audiomentations import AddBackgroundNoise, Reverse
from demo.demo import DEMO_DIR
from tests.utils import fast_autocorr


def test_add_background_noise():
    samples = np.sin(np.linspace(0, 440 * 2 * np.pi, 22500)).astype(np.float32)
    sample_rate = 44100
    augmenter = AddBackgroundNoise(
        sounds_path=os.path.join(DEMO_DIR, "background_noises"),
        min_snr_db=15.0,
        max_snr_db=35.0,
        p=1.0,
    )
    samples_out = augmenter(samples=samples, sample_rate=sample_rate)
    assert not np.allclose(samples, samples_out)
    assert samples_out.dtype == np.float32


def test_background_noise_offset_variation():
    # Check variation in chosen offsets when noise is longer than signal
    np.random.seed(123)
    random.seed(456)
    samples = np.sin(np.linspace(0, 440 * 2 * np.pi, 500)).astype(np.float32)
    sample_rate = 44100
    noise_path = os.path.join(DEMO_DIR, "background_noises", "hens.ogg")
    augmenter = AddBackgroundNoise(sounds_path=noise_path, p=1.0)

    offsets = set()
    for i in range(5):
        augmenter(samples=samples, sample_rate=sample_rate)
        offsets.add(augmenter.parameters["offset"])

    assert len(offsets) >= 4  # we have many different values


def test_background_noise_offset_and_duration():
    # Check the correctness of noise duration and offset
    np.random.seed(44)
    random.seed(55)
    samples = np.sin(np.linspace(0, 440 * 2 * np.pi, 500)).astype(np.float32)
    sample_rate = 2000
    noise_path = os.path.join(DEMO_DIR, "background_noises", "hens.ogg")
    noise, _ = librosa.load(path=noise_path, sr=sample_rate, mono=True)
    noise_duration = noise.shape[-1] / sample_rate
    augmenter = AddBackgroundNoise(
        sounds_path=noise_path,
        min_snr_db=20.0,
        max_snr_db=20.0,
        p=1.0,
    )
    samples_out = augmenter(samples=samples, sample_rate=sample_rate)

    assert augmenter.time_info_arr[0] == pytest.approx(
        noise_duration, abs=1 / sample_rate
    )

    added_noise = samples_out - samples

    best_offset_int = -1
    max_corr_coef = 0.0
    for candidate_offset_int in range(0, noise.shape[-1] - samples.shape[-1]):
        candidate_noise_slice = noise[
            candidate_offset_int : candidate_offset_int + samples.shape[-1]
        ]
        corr_coef = fast_autocorr(added_noise, candidate_noise_slice)
        if corr_coef > max_corr_coef:
            max_corr_coef = corr_coef
            best_offset_int = candidate_offset_int

    assert max_corr_coef > 0.95
    actual_offset = best_offset_int / sample_rate
    assert augmenter.parameters["offset"] == pytest.approx(
        actual_offset, abs=1 / sample_rate
    )


def test_add_background_noise_when_noise_sound_is_too_short():
    sample_rate = 44100
    samples = np.sin(np.linspace(0, 440 * 2 * np.pi, 14 * sample_rate)).astype(
        np.float32
    )
    augmenter = AddBackgroundNoise(
        sounds_path=os.path.join(DEMO_DIR, "background_noises"),
        min_snr_db=15,
        max_snr_db=35,
        p=1.0,
    )
    samples_out = augmenter(samples=samples, sample_rate=sample_rate)
    assert not np.allclose(samples, samples_out)
    assert samples_out.dtype == np.float32


def test_try_add_almost_silent_file():
    samples = np.sin(np.linspace(0, 440 * 2 * np.pi, 30000)).astype(np.float32)
    sample_rate = 48000
    augmenter = AddBackgroundNoise(
        sounds_path=os.path.join(DEMO_DIR, "almost_silent"),
        min_snr_db=15,
        max_snr_db=35,
        p=1.0,
    )
    samples_out = augmenter(samples=samples, sample_rate=sample_rate)
    assert not np.allclose(samples, samples_out)
    assert samples_out.dtype == np.float32


def test_try_add_digital_silence():
    samples = np.sin(np.linspace(0, 440 * 2 * np.pi, 40000)).astype(np.float32)
    sample_rate = 48000
    augmenter = AddBackgroundNoise(
        sounds_path=os.path.join(DEMO_DIR, "digital_silence"),
        min_snr_db=15,
        max_snr_db=35,
        p=1.0,
    )

    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        samples_out = augmenter(samples=samples, sample_rate=sample_rate)

        assert "is too silent to be added as noise" in str(w[-1].message)

    assert np.allclose(samples, samples_out)
    assert samples_out.dtype == np.float32


def test_absolute_option():
    samples = np.sin(np.linspace(0, 440 * 2 * np.pi, 22500)).astype(np.float32)
    sample_rate = 44100
    augmenter = AddBackgroundNoise(
        sounds_path=os.path.join(DEMO_DIR, "background_noises"),
        noise_rms="absolute",
        p=1.0,
    )
    samples_out = augmenter(samples=samples, sample_rate=sample_rate)
    assert not np.allclose(samples, samples_out)


def test_noise_transform():
    np.random.seed(3650)
    random.seed(3650)
    samples = np.sin(np.linspace(0, 440 * 2 * np.pi, 22500)).astype(np.float32)
    sample_rate = 44100
    augmenter = AddBackgroundNoise(
        sounds_path=os.path.join(DEMO_DIR, "background_noises"),
        min_snr_db=3,
        max_snr_db=6,
        p=1.0,
    )
    samples_out_without_transform = augmenter(samples=samples, sample_rate=sample_rate)
    augmenter.freeze_parameters()
    augmenter.noise_transform = Reverse()
    samples_out_with_transform = augmenter(samples=samples, sample_rate=sample_rate)

    assert not np.allclose(samples_out_without_transform, samples_out_with_transform)


def test_validation():
    with pytest.raises(ValueError):
        AddBackgroundNoise(
            sounds_path=os.path.join(DEMO_DIR, "background_noises"),
            min_snr_db=45.0,
            max_snr_db=35.0,
        )
    with pytest.raises(ValueError):
        AddBackgroundNoise(
            sounds_path=os.path.join(DEMO_DIR, "background_noises"),
            min_absolute_rms_db=-1.0,
            max_absolute_rms_db=-5.0,
        )
    with pytest.raises(ValueError):
        AddBackgroundNoise(
            sounds_path=os.path.join(DEMO_DIR, "background_noises"),
            min_absolute_rms_db=-1.0,
            max_absolute_rms_db=10.0,
        )
