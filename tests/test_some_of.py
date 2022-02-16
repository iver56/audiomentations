import os

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from audiomentations import (
    ClippingDistortion,
    AddBackgroundNoise,
    FrequencyMask,
    TimeMask,
    Shift,
    PolarityInversion,
    Gain,
    SpecFrequencyMask,
    SomeOf,
)
from demo.demo import DEMO_DIR


class TestSomeOf:
    def test_right_number_of_transforms_applied(self):
        samples = np.array([0.25, 0.0, 0.1, -0.4], dtype=np.float32)
        sample_rate = 44100
        num_transforms_applied = []
        num_runs = 30
        num_augmenters = 4
        list_transforms = [
            Gain(min_gain_in_db=-12, max_gain_in_db=-6, p=1.0),
            PolarityInversion(p=1.0),
        ]

        for i in range(0, num_augmenters):
            num_transforms_applied_one_augmenter = 0
            for _ in range(num_runs):
                augmenter1 = SomeOf(1, list_transforms)
                augmenter2 = SomeOf((1, 2), list_transforms)
                augmenter3 = SomeOf((2, None), list_transforms)
                augmenter4 = SomeOf((0, None), list_transforms)
                augmenters = [augmenter1, augmenter2, augmenter3, augmenter4]
                augmenter = augmenters[i]
                perturbed_samples = augmenter(samples=samples, sample_rate=sample_rate)

                was_gain_applied = 0.0 < abs(perturbed_samples[0]) < 0.25
                was_polarity_inversion_applied = perturbed_samples[0] < 0.0

                num_transforms_applied_one_iteration = sum(
                    [
                        1 if was_gain_applied else 0,
                        1 if was_polarity_inversion_applied else 0,
                    ]
                )
                num_transforms_applied_one_augmenter += (
                    num_transforms_applied_one_iteration
                )
            num_transforms_applied.append(num_transforms_applied_one_augmenter)
        assert num_transforms_applied[0] / num_runs == 1
        assert 1 < num_transforms_applied[1] / num_runs < 2
        assert num_transforms_applied[2] / num_runs == 2
        assert 1 < num_transforms_applied[3] / num_runs < 2

    def test_freeze_and_unfreeze_all_parameters(self):
        samples = np.array([0.25, 0.0, 0.1, -0.4], dtype=np.float32)
        sample_rate = 44100

        for _ in range(30):
            augmenter = SomeOf((1, None), [Gain(p=1.0), PolarityInversion(p=1.0)])
            perturbed_samples1 = augmenter(samples=samples, sample_rate=sample_rate)
            augmenter.freeze_parameters()
            for transform in augmenter.transforms:
                assert transform.are_parameters_frozen
            perturbed_samples2 = augmenter(samples=samples, sample_rate=sample_rate)

            assert_array_equal(perturbed_samples1, perturbed_samples2)

            augmenter.unfreeze_parameters()
            for transform in augmenter.transforms:
                assert not transform.are_parameters_frozen

    def test_freeze_and_unfreeze_own_parameters(self):
        augmenter = SomeOf(
            (1, None),
            [
                Gain(p=1.0),
                PolarityInversion(p=1.0),
            ],
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

    def test_randomize_all_parameters_and_apply(self):
        samples = 1.0 / np.arange(1, 21, dtype=np.float32)
        sample_rate = 44100

        augmenter = SomeOf(
            (1, None),
            [
                AddBackgroundNoise(
                    sounds_path=os.path.join(DEMO_DIR, "background_noises"),
                    min_snr_in_db=15,
                    max_snr_in_db=35,
                    p=1.0,
                ),
                ClippingDistortion(p=1.0),
                FrequencyMask(min_frequency_band=0.3, max_frequency_band=0.5, p=1.0),
                TimeMask(min_band_part=0.2, max_band_part=0.5, p=1.0),
                Shift(min_fraction=0.5, max_fraction=0.5, p=1.0),
            ],
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

    def test_randomize_only_own_parameters(self):
        samples = 1.0 / np.arange(1, 21, dtype=np.float32)
        sample_rate = 44100

        augmenter = SomeOf(
            (1, None),
            [
                Gain(p=1.0),
                PolarityInversion(p=1.0),
            ],
        )
        augmenter.randomize_parameters(samples, sample_rate, apply_to_children=True)
        own_parameters_before = (augmenter.should_apply, augmenter.transform_indexes)

        parameters = [transform.parameters for transform in augmenter.transforms]

        own_parameters_changed = False
        for i in range(30):
            augmenter.randomize_parameters(
                samples, sample_rate, apply_to_children=False
            )
            if (
                augmenter.should_apply,
                augmenter.transform_indexes,
            ) != own_parameters_before:
                own_parameters_changed = True
                break
        assert own_parameters_changed

        # Check that the children's parameters are still the same
        for transform_parameters, transform in zip(parameters, augmenter.transforms):
            assert transform_parameters == transform.parameters
            assert not transform.are_parameters_frozen

    def test_some_of_spectrogram_magnitude(self):
        spectrogram = np.random.random((128, 128, 2))
        augmenter = SomeOf(
            (1, None),
            [
                SpecFrequencyMask(fill_mode="mean", p=1.0),
                SpecFrequencyMask(fill_mode="constant", p=1.0),
            ],
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

    def test_some_of_spectrogram_magnitude_with_p_0(self):
        spectrogram = np.random.random((128, 128, 2))
        augmenter = SomeOf(
            (1, None),
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

    def test_some_of_spectrogram_magnitude_min_zero(self):
        spectrogram = np.random.random((64, 64, 2))
        augmenter = SomeOf(
            (0, 1),
            [
                SpecFrequencyMask(fill_mode="mean", p=1.0),
                SpecFrequencyMask(fill_mode="constant", p=1.0),
            ],
        )

        for i in range(20):
            augmenter(spectrogram)
