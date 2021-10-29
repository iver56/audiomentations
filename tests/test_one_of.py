import os
import unittest

import numpy as np
from audiomentations import Gain
from numpy.testing import assert_array_equal

from audiomentations.augmentations.transforms import (
    ClippingDistortion,
    AddBackgroundNoise,
    FrequencyMask,
    TimeMask,
    Shift,
    PolarityInversion,
)
from audiomentations import OneOf
from demo.demo import DEMO_DIR


class TestOneOf(unittest.TestCase):
    def test_exactly_one_transform_applied(self):
        samples = np.array([0.25, 0.0, 0.1, -0.4], dtype=np.float32)
        sample_rate = 44100

        for _ in range(30):
            augmenter = OneOf(
                [
                    Gain(min_gain_in_db=-12, max_gain_in_db=-6, p=1.0),
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

    def test_freeze_and_unfreeze_parameters(self):
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
                self.assertTrue(transform.are_parameters_frozen)
            perturbed_samples2 = augmenter(samples=samples, sample_rate=sample_rate)

            assert_array_equal(perturbed_samples1, perturbed_samples2)

            augmenter.unfreeze_parameters()
            for transform in augmenter.transforms:
                self.assertFalse(transform.are_parameters_frozen)

    def test_randomize_parameters_and_apply(self):
        samples = 1.0 / np.arange(1, 21, dtype=np.float32)
        sample_rate = 44100

        augmenter = OneOf(
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
