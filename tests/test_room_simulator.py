import multiprocessing
import os
import pickle
import random

import numpy as np
import pytest

from audiomentations import RoomSimulator

DEBUG = False


def get_sinc_impulse(sample_rate, duration):
    """Create a `duration` seconds chirp from 0Hz to `Nyquist frequency`"""
    n = np.arange(-duration / 2, duration / 2, 1 / sample_rate)

    # Full band sinc impulse centered at half the duration
    samples = 2 * 0.25 * np.sinc(2 * sample_rate / 4 * n)

    return samples.astype(np.float32)


def test_simulate_apply_parity():
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

    assert augmented_samples_simulate == pytest.approx(augmented_samples_apply)


def test_failing_case():
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
def test_multichannel_input(num_channels):
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
def test_input_with_absorption(leave_length_unchanged):
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
def test_input_with_rt60(leave_length_unchanged):
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


@pytest.mark.parametrize("p", [0.0, 1.0])
def test_pickle_before_first_call(p):
    samples = get_sinc_impulse(16000, 0.1)
    augment = RoomSimulator(p=p, use_ray_tracing=False)
    restored = pickle.loads(pickle.dumps(augment))

    random.seed(382)
    expected = augment(samples, sample_rate=16000)
    random.seed(382)
    actual = restored(samples, sample_rate=16000)

    np.testing.assert_array_equal(actual, expected)
    assert restored.parameters == augment.parameters
    assert not restored.are_parameters_frozen


@pytest.mark.parametrize("use_ray_tracing", [False, True])
@pytest.mark.parametrize("calculation_mode", ["absorption", "rt60"])
@pytest.mark.parametrize(
    "sample_rate, num_channels, leave_length_unchanged",
    [(8000, None, True), (16000, 1, False), (44100, 3, False)],
)
def test_pickle_preserves_frozen_output(
    use_ray_tracing, calculation_mode, sample_rate, num_channels, leave_length_unchanged
):
    random.seed(382)
    samples = get_sinc_impulse(sample_rate, 0.1)
    if num_channels is not None:
        samples = np.stack([samples / (i + 1) for i in range(num_channels)])
    augment = RoomSimulator(
        p=1.0,
        use_ray_tracing=use_ray_tracing,
        calculation_mode=calculation_mode,
        leave_length_unchanged=leave_length_unchanged,
    )
    expected = augment(samples, sample_rate=sample_rate)
    augment.freeze_parameters()
    room = augment.room
    parameters = augment.parameters.copy()

    restored = pickle.loads(pickle.dumps(augment))
    actual = restored(samples, sample_rate=sample_rate)

    assert restored.are_parameters_frozen
    assert restored.parameters == parameters
    assert augment.room is room
    assert actual.dtype == samples.dtype
    assert actual.shape[:-1] == samples.shape[:-1]
    expected_length = samples.shape[-1]
    if not leave_length_unchanged:
        expected_length += len(room.rir[0][0]) - 1
    assert actual.shape[-1] == expected_length
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(augment(samples, sample_rate), expected)

    # Preserve the impulse response, not just the output for the initial input.
    other_samples = samples[..., : samples.shape[-1] // 2] * -0.5
    np.testing.assert_array_equal(
        restored(other_samples, sample_rate), augment(other_samples, sample_rate)
    )


def test_pickle_unfrozen_randomization():
    samples = get_sinc_impulse(16000, 0.1)
    random.seed(382)
    augment = RoomSimulator(p=1.0, use_ray_tracing=False, leave_length_unchanged=True)
    augment(samples, sample_rate=16000)
    restored = pickle.loads(pickle.dumps(augment))
    assert not restored.are_parameters_frozen

    for seed, sample_rate in [(383, 16000), (384, 32000)]:
        previous_parameters = restored.parameters.copy()
        random.seed(seed)
        expected = augment(samples, sample_rate)
        random.seed(seed)
        actual = restored(samples, sample_rate)

        assert restored.parameters != previous_parameters
        assert restored.parameters == augment.parameters
        assert restored.room.fs == sample_rate
        assert actual.shape == samples.shape
        np.testing.assert_array_equal(actual, expected)

    restored.freeze_parameters()
    restored = pickle.loads(pickle.dumps(restored))
    restored.unfreeze_parameters()
    previous_parameters = restored.parameters.copy()
    restored(samples, sample_rate=16000)
    assert restored.parameters != previous_parameters


def _apply_in_spawned_process(augment, samples, sample_rate):
    return os.getpid(), augment(samples, sample_rate)


def test_pickle_initialized_transform_with_spawn():
    random.seed(382)
    samples = get_sinc_impulse(16000, 0.1)
    augment = RoomSimulator(p=1.0, use_ray_tracing=True)
    expected = augment(samples, sample_rate=16000)
    augment.freeze_parameters()

    with multiprocessing.get_context("spawn").Pool(1) as pool:
        child_pid, actual = pool.apply_async(
            _apply_in_spawned_process, (augment, samples, 16000)
        ).get(timeout=60)

    assert child_pid != os.getpid()
    np.testing.assert_array_equal(actual, expected)
