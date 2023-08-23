import numpy as np
import pyloudnorm
from numpy.testing import assert_almost_equal

from audiomentations import Gain
from audiomentations.core.gain_compensation import GainCompensation
from audiomentations.core.utils import calculate_rms


class TestGainCompensation:
    def test_same_rms(self):
        samples = np.array([1.0, 0.5, -0.25, -0.125, 0.0], dtype=np.float32)
        sample_rate = 16000

        augment = GainCompensation(
            Gain(min_gain_db=-6, max_gain_db=-6, p=1.0), loudness_metric="same_rms"
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

        augment = GainCompensation(
            Gain(min_gain_db=60, max_gain_db=60, p=1.0), loudness_metric="same_lufs"
        )
        processed_samples = augment(samples=samples, sample_rate=sample_rate)

        meter = pyloudnorm.Meter(sample_rate)  # create BS.1770 meter
        lufs_before = meter.integrated_loudness(samples.transpose())
        lufs_after = meter.integrated_loudness(processed_samples.transpose())
        assert_almost_equal(lufs_after, lufs_before, decimal=6)
        assert processed_samples.dtype == np.float32
