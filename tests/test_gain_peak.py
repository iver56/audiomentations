import warnings

import numpy as np

from audiomentations import GainPeak, Compose


class TestGainPeak:
    def test_gain(self):
        samples = np.array([1.0, 0.5, -0.25, -0.125, 0.0], dtype=np.float32)
        sample_rate = 16000

        augment = Compose([GainPeak(min_gain=1, min_gain_diff=2, max_gain_diff=10, 
            min_peak_relpos=0.3, max_peak_relpos=0.7, p=1.0)])
        processed_samples = augment(samples=samples, sample_rate=sample_rate)
        assert processed_samples.dtype == np.float32

    def test_gain_multichannel(self):
        samples = np.array(
            [[1.0, 0.5, -0.25, -0.125, 0.0], [1.0, 0.5, -0.25, -0.125, 0.0]],
            dtype=np.float32,
        )
        sample_rate = 16000

        augment = Compose([GainPeak(min_gain=1, min_gain_diff=2, max_gain_diff=10, 
            min_peak_relpos=0.3, max_peak_relpos=0.7, p=1.0)])
        processed_samples = augment(samples=samples, sample_rate=sample_rate)
        assert processed_samples.dtype == np.float32

    def test_gain_multichannel_with_wrong_dimension_ordering(self):
        samples = np.array(
            [[1.0, 0.5, -0.25, -0.125, 0.0], [1.0, 0.5, -0.25, -0.125, 0.0]],
            dtype=np.float32,
        ).T
        print(samples.shape)
        sample_rate = 16000

        augment = Compose([GainPeak(min_gain=1, min_gain_diff=2, max_gain_diff=10, 
            min_peak_relpos=0.3, max_peak_relpos=0.7, p=1.0)])

        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")

            processed_samples = augment(samples=samples, sample_rate=sample_rate)

            assert len(w) == 1
            assert "Multichannel audio must have channels first" in str(w[-1].message)
