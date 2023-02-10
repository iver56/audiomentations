import numpy as np
import pyloudnorm
import pytest
from numpy.testing import assert_almost_equal

from audiomentations import Gain
from audiomentations.core.post_gain import PostGain
from audiomentations.core.utils import calculate_rms


class TestPostGain:
    def test_same_rms(self):
        samples = np.array([1.0, 0.5, -0.25, -0.125, 0.0], dtype=np.float32)
        sample_rate = 16000

        augment = PostGain(
            Gain(min_gain_in_db=-6, max_gain_in_db=-6, p=1.0), method="same_rms"
        )
        processed_samples = augment(samples=samples, sample_rate=sample_rate)
        assert_almost_equal(
            calculate_rms(processed_samples),
            calculate_rms(samples),
        )
        assert processed_samples.dtype == np.float32

    def test_same_lufs(self):
        samples = np.random.uniform(low=-0.5, high=0.5, size=(2, 8000)).astype(
            np.float32
        )
        sample_rate = 16000

        augment = PostGain(
            Gain(min_gain_in_db=60, max_gain_in_db=60, p=1.0), method="same_lufs"
        )
        processed_samples = augment(samples=samples, sample_rate=sample_rate)

        meter = pyloudnorm.Meter(sample_rate)  # create BS.1770 meter
        lufs_before = meter.integrated_loudness(samples.transpose())
        lufs_after = meter.integrated_loudness(processed_samples.transpose())
        assert_almost_equal(lufs_after, lufs_before, decimal=6)
        assert processed_samples.dtype == np.float32

    def test_peak_normalize_always(self):
        samples = np.random.uniform(low=-0.5, high=0.5, size=(2, 8000)).astype(
            np.float32
        )
        sample_rate = 16000

        augment = PostGain(
            Gain(min_gain_in_db=-55, max_gain_in_db=-55, p=1.0),
            method="peak_normalize_always",
        )
        processed_samples = augment(samples=samples, sample_rate=sample_rate)

        assert np.amax(np.abs(processed_samples)) == pytest.approx(1.0)
        assert processed_samples.dtype == np.float32
