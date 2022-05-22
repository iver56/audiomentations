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

    def test_simulate_apply_parity(self):
        """
        Tests whether RoomSimulator.apply gives the same result as Roomsimulator.room.simulate() in the 1D case.

        This mainly tests that we took into consideration and compensated about the delays introduced when pyroomacoustics
        computes the room impulse response.

        See:[Create the Room Impulse Response](https://pyroomacoustics.readthedocs.io/en/pypi-release/pyroomacoustics.room.html?highlight=simulate#)
        """

        random.seed(1)
        sample_rate = 16000
        samples = get_sinc_impulse(sample_rate, 10)

        augment = RoomSimulator()
        augmented_samples_apply = augment(samples=samples, sample_rate=sample_rate)
        augment.room.simulate()
        augmented_samples_simulate = augment.room.mic_array.signals.astype(
            np.float32
        ).flatten()

        assert np.all(augmented_samples_apply == augmented_samples_simulate)

    def test_failing_case(self):
        """Failed case which identified a bug where the room created was not rectangular"""
        sample_rate = 16000

        samples = get_sinc_impulse(sample_rate, 10)

        augment = RoomSimulator(
            min_size_x=3.0,
            min_size_y=4.0,
            min_size_z=3.0,
            max_size_x=3.0,
            max_size_y=4.0,
            max_size_z=3.0,
            min_source_x=0.5,
            min_source_y=0.5,
            min_source_z=1.8,
            max_source_x=0.5,
            max_source_y=0.5,
            max_source_z=1.8,
            min_mic_distance=0.1,
            max_mic_distance=0.1,
            p=1.0,
        )
        augment(samples=samples, sample_rate=sample_rate)

    @pytest.mark.parametrize("num_channels", [1, 2, 3])
    def test_multichannel_input(self, num_channels):
        random.seed(1)
        sample_rate = 16000
        samples = get_sinc_impulse(sample_rate, 10)
        n_channels = np.tile(samples, (num_channels, 1))
        augment = RoomSimulator(leave_length_unchanged=True)
        # Setting the seed is important for reproduction
        np.random.seed(1)
        augmented_samples = augment(samples=samples, sample_rate=sample_rate)

        assert augmented_samples.shape == samples.shape

        augment.freeze_parameters()
        np.random.seed(1)
        augmented_n_channels = augment(samples=n_channels, sample_rate=sample_rate)

        assert augmented_n_channels.shape == n_channels.shape

        assert np.allclose(augmented_samples, augmented_n_channels[0])

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
        assert np.isclose(theoretical_rt60, measured_rt60, atol=0.065)
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
            calculation_mode="rt60",
            min_target_rt60=0.3,
            max_target_rt60=0.3,
            leave_length_unchanged=leave_length_unchanged,
        )

        processed_samples = augment(samples=samples, sample_rate=sample_rate)

        # Store a measured rt60 parameter
        theoretical_rt60 = augment.room.rt60_theory()
        measured_rt60 = augment.room.measure_rt60()[0][0]

        # Experimentally set that in this case. Target t60
        # is expected to deviate quite a bit.
        assert np.isclose(0.3, theoretical_rt60, atol=0.05)
        assert np.isclose(theoretical_rt60, measured_rt60, atol=0.065)
        assert processed_samples.dtype == samples.dtype
        assert not np.allclose(processed_samples[: len(samples)], samples)
        assert len(processed_samples.shape) == 1
