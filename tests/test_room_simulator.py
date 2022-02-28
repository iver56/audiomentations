from audiomentations import RoomSimulator
import numpy as np
import random


DEBUG = False


def get_sinc_impulse(sample_rate, duration):
    """Create a `duration` seconds chirp from 0Hz to `nyquist frequency`"""
    n = np.arange(-duration / 2, duration / 2, 1 / sample_rate)

    # Full band sinc impulse centered at half the duration
    samples = 2 * 0.25 * np.sinc(2 * sample_rate / 4 * n)

    return samples.astype(np.float32)


class TestRoomSimulatorTransform:
    def test_input_with_absorption(self):
        random.seed(1)
        sample_rate = 16000
        samples = get_sinc_impulse(sample_rate, 10)

        augment = RoomSimulator(
            p=1.0,
            leave_length_unchanged=True,
        )

        processed_samples = augment(samples=samples, sample_rate=sample_rate)

        # Store a measured rt60 parameter
        theoretical_rt60 = augment.room.rt60_theory()
        measured_rt60 = augment.room.measure_rt60()[0][0]

        # Experimentally set that in this case
        assert np.isclose(theoretical_rt60, measured_rt60, atol=0.015)
        assert processed_samples.dtype == samples.dtype
        assert not np.allclose(processed_samples, samples)
        assert len(processed_samples.shape) == 1

    def test_input_with_rt60(self):
        random.seed(1)
        sample_rate = 16000
        samples = get_sinc_impulse(sample_rate, 10)

        augment = RoomSimulator(
            p=1.0,
            calculate_by_absorption_or_rt60="rt60",
            min_absorption_value_or_rt60=0.06,
            max_absorption_value_or_rt60=0.06,
            leave_length_unchanged=True,
        )

        processed_samples = augment(samples=samples, sample_rate=sample_rate)

        # Store a measured rt60 parameter
        theoretical_rt60 = augment.room.rt60_theory()
        measured_rt60 = augment.room.measure_rt60()[0][0]

        # Experimentally set that in this case. Target t60
        # is expected to deviate quite a bit.
        assert np.isclose(0.06, measured_rt60, atol=0.02)
        assert np.isclose(theoretical_rt60, measured_rt60, atol=0.005)
        assert processed_samples.dtype == samples.dtype
        assert not np.allclose(processed_samples, samples)
        assert len(processed_samples.shape) == 1
