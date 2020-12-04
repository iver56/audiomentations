import os
import unittest

import numpy as np
from numpy.testing import assert_array_equal

from audiomentations.augmentations.transforms import (
    ClippingDistortion,
    AddBackgroundNoise,
    FrequencyMask,
    TimeMask,
    Shift,
)
from audiomentations.core.composition import Compose
from demo.demo import DEMO_DIR


class TestCompose(unittest.TestCase):
    def test_freeze_and_unfreeze_parameters(self):
        samples = np.zeros((20,), dtype=np.float32)
        sample_rate = 44100
        augmenter = Compose(
            [
                AddBackgroundNoise(
                    sounds_path=os.path.join(DEMO_DIR, "background_noises"),
                    min_snr_in_db=15,
                    max_snr_in_db=35,
                    p=1.0,
                ),
                ClippingDistortion(p=0.5),
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

        augmenter = Compose(
            [
                AddBackgroundNoise(
                    sounds_path=os.path.join(DEMO_DIR, "background_noises"),
                    min_snr_in_db=15,
                    max_snr_in_db=35,
                    p=1.0,
                ),
                ClippingDistortion(p=0.5),
                FrequencyMask(min_frequency_band=0.3, max_frequency_band=0.5, p=0.5),
                TimeMask(min_band_part=0.2, max_band_part=0.5, p=0.5),
                Shift(min_fraction=0.5, max_fraction=0.5, p=0.5),
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
            self.assertTrue(transform_parameters == transform.parameters)
            self.assertFalse(transform.are_parameters_frozen)
