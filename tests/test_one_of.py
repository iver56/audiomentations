import os

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from audiomentations import (
    ClippingDistortion,
    AddBackgroundNoise,
    TimeMask,
    Shift,
    PolarityInversion,
    Gain,
    SpecFrequencyMask,
    OneOf,
)
from demo.demo import DEMO_DIR


def test_exactly_one_transform_applied():
    samples = np.array([0.25, 0.0, 0.1, -0.4], dtype=np.float32)
    sample_rate = 44100

    for _ in range(30):
        augmenter = OneOf(
            [
                Gain(min_gain_db=-12, max_gain_db=-6, p=1.0),
                PolarityInversion(p=1.0),
            ]
        )
        perturbed_samples = augmenter(samples=samples, sample_rate=sample_rate)

        was_gain_applied = 0.0 < abs(perturbed_samples[0]) < 0.25
        was_polarity_inversion_applied = perturbed_samples[0] < 0.0

        num_transforms_applied = sum(
            [
                1 if was_gain_applied else 0,
                1 if was_polarity_inversion_applied else 0,
            ]
        )
        assert num_transforms_applied == 1

def test_freeze_and_unfreeze_all_parameters():
    samples = np.array([0.25, 0.0, 0.1, -0.4], dtype=np.float32)
    sample_rate = 44100

    for _ in range(30):
        augmenter = OneOf(
            [
                Gain(p=1.0),
                PolarityInversion(p=1.0),
            ]
        )
        perturbed_samples1 = augmenter(samples=samples, sample_rate=sample_rate)
        augmenter.freeze_parameters()
        for transform in augmenter.transforms:
            assert transform.are_parameters_frozen
        perturbed_samples2 = augmenter(samples=samples, sample_rate=sample_rate)

        assert_array_equal(perturbed_samples1, perturbed_samples2)

        augmenter.unfreeze_parameters()
        for transform in augmenter.transforms:
            assert not transform.are_parameters_frozen

def test_freeze_and_unfreeze_own_parameters():
    augmenter = OneOf(
        [
            Gain(p=1.0),
            PolarityInversion(p=1.0),
        ]
    )
    assert not augmenter.are_parameters_frozen
    for transform in augmenter.transforms:
        assert not transform.are_parameters_frozen

    augmenter.freeze_parameters(apply_to_children=False)
    assert augmenter.are_parameters_frozen
    for transform in augmenter.transforms:
        assert not transform.are_parameters_frozen

    augmenter.unfreeze_parameters(apply_to_children=False)
    assert not augmenter.are_parameters_frozen
    for transform in augmenter.transforms:
        assert not transform.are_parameters_frozen

def test_randomize_all_parameters_and_apply():
    samples = 1.0 / np.arange(1, 21, dtype=np.float32)
    sample_rate = 44100

    augmenter = OneOf(
        [
            AddBackgroundNoise(
                sounds_path=os.path.join(DEMO_DIR, "background_noises"),
                min_snr_db=15,
                max_snr_db=35,
                p=1.0,
            ),
            ClippingDistortion(p=1.0),
            TimeMask(min_band_part=0.2, max_band_part=0.5, p=1.0),
            Shift(min_shift=0.5, max_shift=0.5, p=1.0),
        ]
    )
    augmenter.freeze_parameters()
    augmenter.randomize_parameters(samples=samples, sample_rate=sample_rate)

    parameters = [transform.parameters for transform in augmenter.transforms]

    perturbed_samples1 = augmenter(samples=samples, sample_rate=sample_rate)
    perturbed_samples2 = augmenter(samples=samples, sample_rate=sample_rate)

    assert_array_equal(perturbed_samples1, perturbed_samples2)

    augmenter.unfreeze_parameters()

    for transform_parameters, transform in zip(parameters, augmenter.transforms):
        assert transform_parameters == transform.parameters
        assert not transform.are_parameters_frozen

def test_randomize_only_own_parameters():
    samples = 1.0 / np.arange(1, 21, dtype=np.float32)
    sample_rate = 44100

    augmenter = OneOf(
        [
            Gain(p=1.0),
            PolarityInversion(p=1.0),
        ]
    )
    augmenter.randomize_parameters(samples, sample_rate, apply_to_children=True)
    own_parameters_before = (augmenter.should_apply, augmenter.transform_index)

    parameters = [transform.parameters for transform in augmenter.transforms]

    own_parameters_changed = False
    for i in range(30):
        augmenter.randomize_parameters(
            samples, sample_rate, apply_to_children=False
        )
        if (
            augmenter.should_apply,
            augmenter.transform_index,
        ) != own_parameters_before:
            own_parameters_changed = True
            break
    assert own_parameters_changed

    # Check that the children's parameters are still the same
    for transform_parameters, transform in zip(parameters, augmenter.transforms):
        assert transform_parameters == transform.parameters
        assert not transform.are_parameters_frozen

def test_one_of_spectrogram_magnitude():
    spectrogram = np.random.random((128, 128, 2))
    augmenter = OneOf(
        [
            SpecFrequencyMask(fill_mode="mean", p=1.0),
            SpecFrequencyMask(fill_mode="constant", p=1.0),
        ]
    )

    # Positional argument
    augmented_spectrogram = augmenter(spectrogram)
    assert augmented_spectrogram.shape == spectrogram.shape
    assert augmented_spectrogram.dtype == spectrogram.dtype
    with np.testing.assert_raises(AssertionError):
        assert_array_almost_equal(augmented_spectrogram, spectrogram)

    # Keyword argument
    augmented_spectrogram = augmenter(magnitude_spectrogram=spectrogram)
    assert augmented_spectrogram.shape == spectrogram.shape
    assert augmented_spectrogram.dtype == spectrogram.dtype
    with np.testing.assert_raises(AssertionError):
        assert_array_almost_equal(augmented_spectrogram, spectrogram)

def test_one_of_spectrogram_magnitude_with_p_0():
    spectrogram = np.random.random((128, 128, 2))
    augmenter = OneOf(
        [
            SpecFrequencyMask(fill_mode="mean", p=1.0),
            SpecFrequencyMask(fill_mode="constant", p=1.0),
        ],
        p=0.0,
    )

    # Positional argument
    augmented_spectrogram = augmenter(spectrogram)
    assert augmented_spectrogram.shape == spectrogram.shape
    assert augmented_spectrogram.dtype == spectrogram.dtype
    assert_array_equal(augmented_spectrogram, spectrogram)

    # Keyword argument
    augmented_spectrogram = augmenter(magnitude_spectrogram=spectrogram)
    assert augmented_spectrogram.shape == spectrogram.shape
    assert augmented_spectrogram.dtype == spectrogram.dtype
    assert_array_equal(augmented_spectrogram, spectrogram)
