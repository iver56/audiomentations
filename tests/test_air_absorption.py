DEBUG = False

from audiomentations import AirAbsorption
import numpy as np
import pytest
import scipy
import scipy.signal


def get_chirp_test(sample_rate, duration):
    """Create a `duration` seconds chirp from 0Hz to `nyquist frequency`"""
    n = np.arange(0, duration, 1 / sample_rate)
    samples = scipy.signal.chirp(n, 0, duration, sample_rate // 2, method="linear")
    return samples.astype(np.float32)


class TestAirAbsorptionTransform:
    @pytest.mark.parametrize("temperature", [10, 20])
    @pytest.mark.parametrize("humidity", [30, 50, 70, 90])
    @pytest.mark.parametrize("distance", [5, 10, 20, 100])
    @pytest.mark.parametrize("sample_rate", [8000, 16000, 48000])
    def test_input_shapes(self, temperature, humidity, distance, sample_rate):
        np.random.seed(1)

        samples = get_chirp_test(sample_rate, 10)

        augment = AirAbsorption(
            min_temperature=temperature,
            max_temperature=temperature,
            min_humidity=humidity,
            max_humidity=humidity,
            min_distance=distance,
            max_distance=distance,
        )

        # Test 1D case
        processed_samples = augment(samples, sample_rate=sample_rate)
        assert processed_samples.shape == samples.shape
        assert processed_samples.dtype == np.float32

        # Test 2D case
        samples = np.tile(samples, (2, 1))
        processed_samples = augment(samples, sample_rate=sample_rate)
        assert processed_samples.shape == samples.shape
        assert processed_samples.dtype == np.float32
