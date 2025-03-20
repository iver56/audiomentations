from copy import deepcopy

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from audiomentations import AdjustDuration, Gain, RepeatPart, Shift
from audiomentations.core.utils import get_crossfade_mask_pair
from tests.test_peaking_filter import get_chirp_test


def adapt_ndim(samples, ndim):
    if samples.ndim < ndim:
        samples = samples[np.newaxis, :]
    return samples


def assert_high_frequency_energy_absence(
    audio_array,
    sample_rate,
    frequency_threshold: float = 10_000.0,
    energy_threshold: float = 0.15,
):
    spectrum = np.fft.rfft(audio_array)
    freqs = np.fft.rfftfreq(len(audio_array), 1 / sample_rate)
    assert np.all(
        np.abs(spectrum[freqs > frequency_threshold]) < energy_threshold
    ), "Detected energy in frequencies above the given frequency_threshold!"


def test_replace_one_repeat():
    augment = RepeatPart(mode="replace", crossfade_duration=0.0, p=1.0)
    augment.parameters = {
        "should_apply": True,
        "part_num_samples": 3,
        "repeats": 1,
        "part_start_index": 1,
    }
    augment.freeze_parameters()

    for ndim in (1, 2):
        samples = np.array(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            dtype=np.float32,
        )
        samples = adapt_ndim(samples, ndim)
        processed_samples = augment(samples=samples, sample_rate=4000)
        target_samples = np.array(
            [0.0, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 1.0],
            dtype=np.float32,
        )
        target_samples = adapt_ndim(target_samples, ndim)
        assert_array_almost_equal(processed_samples, target_samples)
        assert processed_samples.shape == samples.shape
        assert processed_samples.dtype == np.float32


def test_replace_one_repeat_transformed():
    augment = RepeatPart(
        mode="replace",
        crossfade_duration=0.0,
        part_transform=Gain(min_gain_db=-6.0, max_gain_db=-6.0, p=1.0),
        p=1.0,
    )
    part_gain_factor = 0.5011872336272722
    augment.part_transform.parameters = {
        "should_apply": True,
        "amplitude_ratio": part_gain_factor,  # -6 dB
    }
    augment.parameters = {
        "should_apply": True,
        "part_num_samples": 3,
        "repeats": 1,
        "part_start_index": 1,
    }
    augment.freeze_parameters()

    for ndim in (1, 2):
        samples = np.array(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            dtype=np.float32,
        )
        samples = adapt_ndim(samples, ndim)

        processed_samples = augment(samples=samples, sample_rate=4000)

        target_samples = np.array(
            [
                0.0,
                0.1,
                0.2,
                0.3,
                0.1 * part_gain_factor,
                0.2 * part_gain_factor,
                0.3 * part_gain_factor,
                0.7,
                0.8,
                0.9,
                1.0,
            ],
            dtype=np.float32,
        )
        target_samples = adapt_ndim(target_samples, ndim)
        assert_array_almost_equal(processed_samples, target_samples)
        assert processed_samples.shape == samples.shape
        assert processed_samples.dtype == np.float32


def test_freeze_and_unfreeze_part_transform_parameters():
    augment = RepeatPart(
        min_part_duration=0.1,
        max_part_duration=0.2,
        part_transform=Gain(p=1.0),
        crossfade_duration=0.0,
        p=1.0,
    )
    dummy_samples = np.zeros(40, dtype=np.float32)
    augment(dummy_samples, 40)
    params1 = deepcopy(augment.part_transform.parameters)
    augment.freeze_parameters()
    augment(dummy_samples, 40)
    params2 = deepcopy(augment.part_transform.parameters)
    assert params1 == params2

    augment.unfreeze_parameters()
    augment(dummy_samples, 40)
    params3 = deepcopy(augment.part_transform.parameters)
    assert params3 != params2


def test_replace_one_repeat_near_end():
    augment = RepeatPart(mode="replace", crossfade_duration=0.0, p=1.0)
    augment.parameters = {
        "should_apply": True,
        "part_num_samples": 3,
        "repeats": 1,
        "part_start_index": 7,
    }
    augment.freeze_parameters()

    for ndim in (1, 2):
        samples = np.array(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            dtype=np.float32,
        )
        samples = adapt_ndim(samples, ndim)
        processed_samples = augment(samples=samples, sample_rate=4000)
        target_samples = np.array(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.7],
            dtype=np.float32,
        )
        target_samples = adapt_ndim(target_samples, ndim)
        assert_array_almost_equal(processed_samples, target_samples)
        assert processed_samples.shape == samples.shape
        assert processed_samples.dtype == np.float32


def test_replace_two_repeats():
    augment = RepeatPart(mode="replace", crossfade_duration=0.0, p=1.0)
    augment.parameters = {
        "should_apply": True,
        "part_num_samples": 3,
        "repeats": 2,
        "part_start_index": 1,
    }
    augment.freeze_parameters()

    for ndim in (1, 2):
        samples = np.array(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            dtype=np.float32,
        )
        samples = adapt_ndim(samples, ndim)
        processed_samples = augment(samples=samples, sample_rate=4000)
        target_samples = np.array(
            [0.0, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 1.0],
            dtype=np.float32,
        )
        target_samples = adapt_ndim(target_samples, ndim)
        assert_array_almost_equal(processed_samples, target_samples)
        assert processed_samples.shape == samples.shape
        assert processed_samples.dtype == np.float32


def test_replace_many_repeats_exceed_input_length():
    augment = RepeatPart(mode="replace", crossfade_duration=0.0, p=1.0)
    augment.parameters = {
        "should_apply": True,
        "part_num_samples": 3,
        "repeats": 9,
        "part_start_index": 1,
    }
    augment.freeze_parameters()

    for ndim in (1, 2):
        samples = np.array(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            dtype=np.float32,
        )
        samples = adapt_ndim(samples, ndim)
        processed_samples = augment(samples=samples, sample_rate=4000)
        target_samples = np.array(
            [0.0, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1],
            dtype=np.float32,
        )
        target_samples = adapt_ndim(target_samples, ndim)
        assert_array_almost_equal(processed_samples, target_samples)
        assert processed_samples.shape == samples.shape
        assert processed_samples.dtype == np.float32


def test_replace_many_transformed_repeats_exceed_input_length():
    augment = RepeatPart(
        mode="replace",
        crossfade_duration=0.0,
        part_transform=Gain(min_gain_db=-6.0, max_gain_db=-6.0, p=1.0),
        p=1.0,
    )
    part_gain_factor = 0.5011872336272722
    augment.part_transform.parameters = {
        "should_apply": True,
        "amplitude_ratio": part_gain_factor,  # -6 dB
    }
    augment.parameters = {
        "should_apply": True,
        "part_num_samples": 3,
        "repeats": 9,
        "part_start_index": 1,
    }
    augment.freeze_parameters()

    for ndim in (1, 2):
        samples = np.array(
            [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            dtype=np.float32,
        )
        samples = adapt_ndim(samples, ndim)
        processed_samples = augment(samples=samples, sample_rate=4000)
        target_samples = np.array(
            [0.0, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1],
            dtype=np.float32,
        )
        target_samples[4:] *= part_gain_factor
        target_samples = adapt_ndim(target_samples, ndim)
        assert_array_almost_equal(processed_samples, target_samples)
        assert processed_samples.shape == samples.shape
        assert processed_samples.dtype == np.float32


def test_insert_one_repeat():
    augment = RepeatPart(mode="insert", crossfade_duration=0.0, p=1.0)
    augment.parameters = {
        "should_apply": True,
        "part_num_samples": 3,
        "repeats": 1,
        "part_start_index": 1,
    }
    augment.freeze_parameters()

    samples = np.array(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float32
    )
    processed_samples = augment(samples=samples, sample_rate=4000)
    assert_array_almost_equal(
        processed_samples,
        np.array(
            [0.0, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            dtype=np.float32,
        ),
    )
    assert processed_samples.dtype == np.float32


def test_insert_two_repeats():
    augment = RepeatPart(mode="insert", crossfade_duration=0.0, p=1.0)
    augment.parameters = {
        "should_apply": True,
        "part_num_samples": 3,
        "repeats": 2,
        "part_start_index": 0,
    }
    augment.freeze_parameters()

    samples = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)
    processed_samples = augment(samples=samples, sample_rate=4000)
    assert_array_almost_equal(
        processed_samples,
        np.array(
            [0.0, 0.1, 0.2, 0.0, 0.1, 0.2, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            dtype=np.float32,
        ),
    )
    assert processed_samples.dtype == np.float32


def test_insert_two_elongated_repeats():
    augment = RepeatPart(
        mode="insert",
        crossfade_duration=0.0,
        part_transform=AdjustDuration(duration_samples=6, p=1.0),
        p=1.0,
    )
    augment.parameters = {
        "should_apply": True,
        "part_num_samples": 3,
        "repeats": 2,
        "part_start_index": 0,
    }
    augment.freeze_parameters()

    samples = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)
    processed_samples = augment(samples=samples, sample_rate=4000)
    assert_array_almost_equal(
        processed_samples,
        np.array(
            [
                0.0,
                0.1,
                0.2,
                0.0,  # first repetition starts here
                0.1,
                0.2,
                0.0,  # start of zero padding
                0.0,
                0.0,  # end of zero padding
                0.0,  # second repetition starts here
                0.1,
                0.2,
                0.0,  # start of zero padding
                0.0,
                0.0,  # end of zero padding
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
            ],
            dtype=np.float32,
        ),
    )
    assert processed_samples.dtype == np.float32


def test_insert_two_repeats_with_crossfading():
    augment = RepeatPart(mode="insert", crossfade_duration=0.005, p=1.0)
    augment.parameters = {
        "should_apply": True,
        "part_num_samples": int(0.25 * 44100),
        "repeats": 2,
        "part_start_index": 150,
    }
    augment.freeze_parameters()

    sample_rate = 44100
    samples = get_chirp_test(sample_rate, duration=2) * 0.3
    processed_samples = augment(samples=samples, sample_rate=sample_rate)

    assert processed_samples.dtype == np.float32


def test_insert_two_repeats_with_crossfading_and_part_transform():
    # The purpose of this test is to cover two different cases:
    # 1) The fade from the last part to the tail is correlated (use equal-gain fade)
    # 2) The fade from the last part to the tail is uncorrelated (use equal-energy fade)
    for part_transform in [Gain(p=0.0), Shift(p=1.0)]:
        augment = RepeatPart(
            mode="insert",
            crossfade_duration=0.005,
            part_transform=part_transform,
            p=1.0,
        )
        augment.parameters = {
            "should_apply": True,
            "part_num_samples": int(0.25 * 44100),
            "repeats": 2,
            "part_start_index": 150,
        }
        augment.freeze_parameters()

        sample_rate = 44100
        samples = get_chirp_test(sample_rate, duration=2) * 0.3
        processed_samples = augment(samples=samples, sample_rate=sample_rate)

        assert processed_samples.dtype == np.float32


def test_replace_mode_two_repeats_with_crossfading():
    augment = RepeatPart(mode="replace", crossfade_duration=0.005, p=1.0)
    part_num_samples = int(0.25 * 44100)
    augment.parameters = {
        "should_apply": True,
        "part_num_samples": part_num_samples,
        "repeats": 2,
        "part_start_index": 142,
    }
    augment.freeze_parameters()

    sample_rate = 44100
    amplitude = 0.3
    samples = get_chirp_test(sample_rate, duration=2) * amplitude
    processed_samples = augment(samples=samples, sample_rate=sample_rate)

    assert processed_samples.dtype == np.float32

    augment_without_crossfade = RepeatPart(
        mode="replace", crossfade_duration=0.000, p=1.0
    )
    augment_without_crossfade.parameters = augment.parameters
    augment_without_crossfade.freeze_parameters()
    processed_samples_without_crossfade = augment_without_crossfade(
        samples=samples, sample_rate=sample_rate
    )

    idx0 = augment.parameters["part_start_index"] + part_num_samples - 1
    idx1 = idx0 + 1
    seam_impulse_magnitude_without_crossfade = abs(
        processed_samples_without_crossfade[..., idx1]
        - processed_samples_without_crossfade[..., idx0]
    )
    assert seam_impulse_magnitude_without_crossfade > amplitude  # audible click!
    seam_impulse_magnitude_with_crossfade = abs(
        processed_samples[..., idx1] - processed_samples[..., idx0]
    )
    assert seam_impulse_magnitude_with_crossfade < 0.11 * amplitude  # smooth

    assert processed_samples.shape == processed_samples_without_crossfade.shape


def test_replace_mode_repeats_with_crossfading_at_end():
    # Test that it doesn't crash if the end of the array is in the middle of
    # the crossfade of the last part (in replace mode)
    augment = RepeatPart(mode="replace", crossfade_duration=0.124, p=1.0)
    part_num_samples = int(0.29 * 44100)
    augment.parameters = {
        "should_apply": True,
        "part_num_samples": part_num_samples,
        "repeats": 5,
        "part_start_index": int(0.22 * 44100),
    }
    augment.freeze_parameters()

    sample_rate = 44100
    amplitude = 0.3
    samples = get_chirp_test(sample_rate, duration=2) * amplitude
    processed_samples = augment(samples=samples, sample_rate=sample_rate)
    assert processed_samples.shape == samples.shape


def test_replace_mode_many_repeats_with_crossfading():
    # Test that it doesn't fade in some of the original audio at the end when there
    # enough repetitions to fill the whole array when in replace mode.
    augment = RepeatPart(mode="replace", crossfade_duration=0.124, p=1.0)
    part_num_samples = int(0.27 * 44100)
    augment.parameters = {
        "should_apply": True,
        "part_num_samples": part_num_samples,
        "repeats": 20,
        "part_start_index": int(0.22 * 44100),
    }
    augment.freeze_parameters()

    sample_rate = 44100
    amplitude = 0.3
    samples = get_chirp_test(sample_rate, duration=2) * amplitude
    processed_samples = augment(samples=samples, sample_rate=sample_rate)
    assert processed_samples.shape == samples.shape

    # The very last part should not have any crossfade into the original signal,
    # where there is only high-freq audio.
    assert_high_frequency_energy_absence(
        processed_samples[int(1.9 * 44100) :], sample_rate
    )


def test_invalid_parameters():
    with pytest.raises(ValueError):
        RepeatPart(crossfade_duration=-0.0001)
    with pytest.raises(ValueError):
        RepeatPart(crossfade_duration=0.0001)
    with pytest.raises(ValueError):
        RepeatPart(min_part_duration=0.0001)
    with pytest.raises(ValueError):
        RepeatPart(min_repeats=0, max_repeats=1)
    with pytest.raises(ValueError):
        RepeatPart(min_repeats=2, max_repeats=1)
    with pytest.raises(ValueError):
        RepeatPart(min_part_duration=0.5, max_part_duration=0.2)
    with pytest.raises(ValueError):
        RepeatPart(min_part_duration=0.01, crossfade_duration=0.1)
    with pytest.raises(ValueError):
        RepeatPart(mode="append")


def test_warn_too_short_input():
    augmenter = RepeatPart(
        mode="replace",
        min_part_duration=0.1,
        max_part_duration=0.1,
        crossfade_duration=0.0,
        p=1.0,
    )
    samples = np.random.uniform(low=-0.5, high=0.5, size=(100,)).astype(np.float32)
    with pytest.warns(
        UserWarning,
        match=(
            "The input sound is not long enough for applying the RepeatPart"
            " transform with the current parameters."
        ),
    ):
        processed_samples = augmenter(samples, sample_rate=10_000)

    assert_array_equal(processed_samples, samples)


def test_very_short_crossfade_mask_pair():
    fade_in, fade_out = get_crossfade_mask_pair(2)
    assert_array_almost_equal(fade_in, np.array([0.0, 1.0], dtype=np.float32))
    assert_array_almost_equal(fade_out, np.array([1.0, 0.0], dtype=np.float32))
