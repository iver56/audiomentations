from audiomentations import RoomSimulator
import numpy as np
import random
import unittest.mock
import pytest

DEBUG = False


def get_sinc_impulse(sample_rate, duration):
    """Create a `duration` seconds chirp from 0Hz to `nyquist frequency`"""
    n = np.arange(-duration / 2, duration / 2, 1 / sample_rate)

    # Full band sinc impulse centered at half the duration
    samples = 2 * 0.25 * np.sinc(2 * sample_rate / 4 * n)

    return samples.astype(np.float32)


class TestRoomSimulatorTransform:
    def test_pyroomacoustics_not_found(self):
        """
        Test raising ImportError when pyroomacoustics is not found on
        randomize_parameters
        """
        random.seed(1)
        sample_rate = 16000
        samples = get_sinc_impulse(sample_rate, 10)
        with pytest.raises(ImportError):
            with unittest.mock.patch.dict("sys.modules", {"pyroomacoustics": None}):
                augment = RoomSimulator()
                augment(samples=samples, sample_rate=sample_rate)
                augment.freeze_parameters()

        """
        Test raising ImportError when pyroomacoustics is not found on apply
        """
        with pytest.raises(ImportError):
            augment = RoomSimulator()
            augment(samples=samples, sample_rate=sample_rate)
            augment.freeze_parameters()
            with unittest.mock.patch.dict("sys.modules", {"pyroomacoustics": None}):
                augment(samples=samples, sample_rate=sample_rate)

    @pytest.mark.parametrize("num_channels", [1, 2, 3])
    def test_multichannel_input(self, num_channels):
        random.seed(1)
        sample_rate = 16000
        samples = get_sinc_impulse(sample_rate, 10)
        n_channels = np.tile(samples, (num_channels, 1))
        augment = RoomSimulator()
        augment.freeze_parameters()
        augmented_samples = augment(samples=samples, sample_rate=sample_rate)
        augmented_n_channels = augment(samples=n_channels, sample_rate=sample_rate)

        assert np.allclose(augmented_samples, augmented_n_channels)

    @pytest.mark.parametrize("leave_length_unchanged", [True, False])
    def test_input_with_absorption(self, leave_length_unchanged):
        random.seed(1)
        sample_rate = 16000
        samples = get_sinc_impulse(sample_rate, 10)

        augment = RoomSimulator(
            p=1.0,
            leave_length_unchanged=leave_length_unchanged,
        )

        processed_samples = augment(samples=samples, sample_rate=sample_rate)

        # Store a measured rt60 parameter
        theoretical_rt60 = augment.room.rt60_theory()
        measured_rt60 = augment.room.measure_rt60()[0][0]

        # Experimentally set that in this case
        assert np.isclose(theoretical_rt60, measured_rt60, atol=0.015)
        assert processed_samples.dtype == samples.dtype
        assert not np.allclose(processed_samples[: len(samples)], samples)
        assert len(processed_samples.shape) == 1

    @pytest.mark.parametrize("leave_length_unchanged", [True, False])
    def test_input_with_rt60(self, leave_length_unchanged):
        random.seed(1)
        sample_rate = 16000
        samples = get_sinc_impulse(sample_rate, 10)

        augment = RoomSimulator(
            p=1.0,
            calculate_by_absorption_or_rt60="rt60",
            min_absorption_value_or_rt60=0.06,
            max_absorption_value_or_rt60=0.06,
            leave_length_unchanged=leave_length_unchanged,
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
        assert not np.allclose(processed_samples[: len(samples)], samples)
        assert len(processed_samples.shape) == 1
