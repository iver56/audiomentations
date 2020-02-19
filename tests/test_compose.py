import os
import unittest

import numpy as np
from numpy.testing import assert_array_equal

from audiomentations.augmentations.transforms import (
    ClippingDistortion,
    AddBackgroundNoise,
)
from audiomentations.core.composition import Compose
from demo.demo import DEMO_DIR


class TestCompose(unittest.TestCase):
    def test_freeze_and_unfreeze_parameters(self):
        samples = np.zeros((20,), dtype=np.float32)
        sample_rate = 16000
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
