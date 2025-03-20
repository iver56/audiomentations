import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from audiomentations import Shift


def test_shift_fraction():
    samples = np.array([1.0, 0.5, 0.25, 0.125], dtype=np.float32)
    sample_rate = 16000

    forward_augmenter = Shift(min_shift=0.5, max_shift=0.5, fade_duration=0.0, p=1.0)
    forward_shifted_samples = forward_augmenter(
        samples=samples, sample_rate=sample_rate
    )
    assert_almost_equal(
        forward_shifted_samples, np.array([0.25, 0.125, 1.0, 0.5], dtype=np.float32)
    )
    assert forward_shifted_samples.dtype == np.float32
    assert len(forward_shifted_samples) == 4

    backward_augmenter = Shift(
        min_shift=-0.25, max_shift=-0.25, fade_duration=0.0, p=1.0
    )
    backward_shifted_samples = backward_augmenter(
        samples=samples, sample_rate=sample_rate
    )
    assert_almost_equal(
        backward_shifted_samples,
        np.array([0.5, 0.25, 0.125, 1.0], dtype=np.float32),
    )
    assert backward_shifted_samples.dtype == np.float32
    assert len(forward_shifted_samples) == 4


def test_shift_samples():
    samples = np.array([1.0, 0.5, 0.25, 0.125], dtype=np.float32)
    sample_rate = 16000

    forward_augmenter = Shift(
        min_shift=1, max_shift=1, shift_unit="samples", fade_duration=0.0, p=1.0
    )
    forward_shifted_samples = forward_augmenter(
        samples=samples, sample_rate=sample_rate
    )
    assert_almost_equal(
        forward_shifted_samples, np.array([0.125, 1.0, 0.5, 0.25], dtype=np.float32)
    )
    assert forward_shifted_samples.dtype == np.float32
    assert len(forward_shifted_samples) == 4


def test_shift_seconds():
    samples = np.array([1.0, 0.5, 0.25, 0.125], dtype=np.float32)
    sample_rate = 2

    forward_augmenter = Shift(
        min_shift=1.0, max_shift=1.0, shift_unit="seconds", fade_duration=0.0, p=1.0
    )
    forward_shifted_samples = forward_augmenter(
        samples=samples, sample_rate=sample_rate
    )
    assert_almost_equal(
        forward_shifted_samples, np.array([0.25, 0.125, 1.0, 0.5], dtype=np.float32)
    )
    assert forward_shifted_samples.dtype == np.float32
    assert len(forward_shifted_samples) == 4


def test_shift_without_rollover():
    samples = np.array([1.0, 0.5, 0.25, 0.125], dtype=np.float32)
    sample_rate = 16000

    forward_augmenter = Shift(
        min_shift=0.5, max_shift=0.5, rollover=False, fade_duration=0.0, p=1.0
    )
    forward_shifted_samples = forward_augmenter(
        samples=samples, sample_rate=sample_rate
    )
    assert_almost_equal(
        forward_shifted_samples, np.array([0.0, 0.0, 1.0, 0.5], dtype=np.float32)
    )
    assert forward_shifted_samples.dtype == np.float32
    assert len(forward_shifted_samples) == 4

    backward_augmenter = Shift(
        min_shift=-0.25, max_shift=-0.25, rollover=False, fade_duration=0.0, p=1.0
    )
    backward_shifted_samples = backward_augmenter(
        samples=samples, sample_rate=sample_rate
    )
    assert_almost_equal(
        backward_shifted_samples,
        np.array([0.5, 0.25, 0.125, 0.0], dtype=np.float32),
    )
    assert backward_shifted_samples.dtype == np.float32
    assert len(forward_shifted_samples) == 4


def test_shift_multichannel():
    samples = np.array(
        [[0.75, 0.5, -0.25, -0.125], [0.9, 0.5, -0.25, -0.125]], dtype=np.float32
    )
    sample_rate = 4000

    augment = Shift(min_shift=0.5, max_shift=0.5, fade_duration=0.0, p=1.0)
    processed_samples = augment(samples=samples, sample_rate=sample_rate)

    assert_almost_equal(
        processed_samples,
        np.array(
            [[-0.25, -0.125, 0.75, 0.5], [-0.25, -0.125, 0.9, 0.5]],
            dtype=np.float32,
        ),
    )
    assert processed_samples.dtype == np.float32


def test_shift_without_rollover_multichannel():
    samples = np.array(
        [[0.75, 0.5, -0.25, -0.125], [0.9, 0.5, -0.25, -0.125]], dtype=np.float32
    )
    sample_rate = 4000

    augment = Shift(
        min_shift=0.5, max_shift=0.5, rollover=False, fade_duration=0.0, p=1.0
    )
    processed_samples = augment(samples=samples, sample_rate=sample_rate)

    assert_almost_equal(
        processed_samples,
        np.array([[0.0, 0.0, 0.75, 0.5], [0.0, 0.0, 0.9, 0.5]], dtype=np.float32),
    )
    assert processed_samples.dtype == np.float32


def test_shift_fade():
    samples = np.array(
        [[1.0, 2.0, 3.0, 4.0, 5.0], [-1.0, -2.0, -3.0, -4.0, -5.0]],
        dtype=np.float32,
    )
    sample_rate = 4000

    augment = Shift(
        min_shift=0.5,
        max_shift=0.5,
        rollover=False,
        fade_duration=0.00075,  # 0.00075 * 4000 = 3
        p=1.0,
    )
    processed_samples = augment(samples=samples, sample_rate=sample_rate)

    assert processed_samples == pytest.approx(
        np.array(
            [[0.0, 0.0, 0.0, 1.4067, 3.0], [0.0, 0.0, 0.0, -1.4067, -3.0]],
            dtype=np.float32,
        ),
        abs=0.01,
    )


def test_shift_fade_rollover():
    samples = np.array(
        [[1.0, 2.0, 3.0, 4.0, 5.0], [-1.0, -2.0, -3.0, -4.0, -5.0]],
        dtype=np.float32,
    )
    sample_rate = 4000

    augment = Shift(
        min_shift=0.5,
        max_shift=0.5,
        rollover=True,
        fade_duration=0.00075,  # 0.00075 * 4000 = 3
        p=1.0,
    )
    processed_samples = augment(samples=samples, sample_rate=sample_rate)
    assert processed_samples == pytest.approx(
        np.array(
            [[2.81, 0.0, 0, 1.407, 3.0], [-2.81, 0.0, 0, -1.407, -3.0]],
            dtype=np.float32,
        ),
        abs=0.01,
    )


def test_shift_fade_rollover_2():
    samples = np.array(
        [[1.0, 2.0, 3.0, 4.0, 5.0], [-1.0, -2.0, -3.0, -4.0, -5.0]],
        dtype=np.float32,
    )
    sample_rate = 4000

    augment = Shift(
        min_shift=-0.5,
        max_shift=-0.5,
        rollover=True,
        fade_duration=0.00075,  # 0.00075 * 4000 = 3
        p=1.0,
    )
    processed_samples = augment(samples=samples, sample_rate=sample_rate)
    assert processed_samples == pytest.approx(
        np.array(
            [[3.0, 2.81, 0.0, 0.0, 1.407], [-3.0, -2.81, 0.0, -0.0, -1.407]],
            dtype=np.float32,
        ),
        abs=0.01,
    )


def test_shift_fade_rollover_3():
    samples = np.array(
        [[1.0, 2.0, 3.0, 4.0, 5.0], [-1.0, -2.0, -3.0, -4.0, -5.0]],
        dtype=np.float32,
    )
    sample_rate = 4000

    augment = Shift(
        min_shift=-0.5,
        max_shift=-0.5,
        rollover=True,
        fade_duration=1.0,
        p=1.0,
    )
    processed_samples = augment(samples=samples, sample_rate=sample_rate)
    assert processed_samples == pytest.approx(
        np.array(
            [
                [
                    3.0023373e-06,
                    1.0004364e-06,
                    0.0000000e00,
                    0.0000000e00,
                    5.0030241e-07,
                ],
                [
                    -3.0023373e-06,
                    -1.0004364e-06,
                    -0.0000000e00,
                    -0.0000000e00,
                    -5.0030241e-07,
                ],
            ],
            dtype=np.float32,
        ),
    )


def test_freeze_shift_with_different_sample_rates():
    samples1 = np.array([1.0, 0.5, 0.25, 0.125], dtype=np.float32)
    samples2 = np.array([1.0, 0.5, 0.25, 0.125, 0.3, 0.4, 0.5, 0.2], dtype=np.float32)
    sample_rate_1 = 1
    sample_rate_2 = 2

    augment = Shift(
        min_shift=0.5, max_shift=0.5, rollover=False, fade_duration=0.0, p=1.0
    )
    augment.randomize_parameters(samples1, sample_rate_1)
    augment.freeze_parameters()
    shifted_samples1 = augment(samples1, sample_rate_1)
    shifted_samples2 = augment(samples2, sample_rate_2)

    num_samples1_shifted = np.count_nonzero(shifted_samples1 == 0)
    num_samples2_shifted = np.count_nonzero(shifted_samples2 == 0)

    assert num_samples1_shifted == 2
    assert num_samples2_shifted == 4


def test_invalid_parameters():
    with pytest.raises(ValueError):
        Shift(min_shift=-1.5)
    with pytest.raises(ValueError):
        Shift(min_shift=1.5)
    with pytest.raises(ValueError):
        Shift(max_shift=-1.5)
    with pytest.raises(ValueError):
        Shift(max_shift=1.5)
    with pytest.raises(ValueError):
        Shift(min_shift=0.2, max_shift=0.1)
    with pytest.raises(ValueError):
        Shift(fade_duration=0.000001)
    with pytest.raises(ValueError):
        Shift(fade_duration=-1337)
    with pytest.raises(ValueError):
        Shift(shift_unit="lightyears")
