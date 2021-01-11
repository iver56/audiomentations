import unittest

import numpy as np
from numpy.testing import assert_array_equal

from audiomentations.augmentations.spectrogram_transforms import (
    SpecChannelShuffle,
    SpecFrequencyMask,
)
from audiomentations.core.composition import SpecCompose


class TestSpecCompose(unittest.TestCase):
    def test_freeze_and_unfreeze_parameters(self):
        spectrogram = np.zeros((256, 256), dtype=np.float32)
        augmenter = SpecCompose(
            [
                SpecChannelShuffle(),
                SpecFrequencyMask(),
            ]
        )
        perturbed_samples1 = augmenter(magnitude_spectrogram=spectrogram)
        augmenter.freeze_parameters()
        for transform in augmenter.transforms:
            self.assertTrue(transform.are_parameters_frozen)
        perturbed_samples2 = augmenter(magnitude_spectrogram=spectrogram)

        assert_array_equal(perturbed_samples1, perturbed_samples2)

        augmenter.unfreeze_parameters()
        for transform in augmenter.transforms:
            self.assertFalse(transform.are_parameters_frozen)

    def test_randomize_parameters_and_apply(self):
        spectrogram = np.zeros((256, 256), dtype=np.float32)
        augmenter = SpecCompose(
            [
                SpecChannelShuffle(),
                SpecFrequencyMask(),
            ]
        )
        augmenter.freeze_parameters()
        augmenter.randomize_parameters(magnitude_spectrogram=spectrogram)

        parameters = [transform.parameters for transform in augmenter.transforms]

        perturbed_samples1 = augmenter(magnitude_spectrogram=spectrogram)
        perturbed_samples2 = augmenter(magnitude_spectrogram=spectrogram)

        assert_array_equal(perturbed_samples1, perturbed_samples2)

        augmenter.unfreeze_parameters()

        for transform_parameters, transform in zip(parameters, augmenter.transforms):
            self.assertTrue(transform_parameters == transform.parameters)
            self.assertFalse(transform.are_parameters_frozen)
