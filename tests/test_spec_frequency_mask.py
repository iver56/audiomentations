import os
import unittest

import librosa
import numpy as np

from audiomentations.augmentations.spectrogram_transforms import SpecFrequencyMask
from demo.demo import DEMO_DIR


class TestSpecFrequencyMask(unittest.TestCase):
    def test_fill_zeros(self):
        samples, sample_rate = librosa.load(
            os.path.join(DEMO_DIR, "acoustic_guitar_0.wav")
        )
        magnitude_spectrogram = librosa.feature.melspectrogram(
            y=samples, sr=sample_rate
        )

        mask_fraction = 0.05
        transform = SpecFrequencyMask(
            fill_mode="constant",
            fill_constant=0.0,
            min_mask_fraction=mask_fraction,
            max_mask_fraction=mask_fraction,
            p=1.0,
        )
        augmented_spectrogram = transform(magnitude_spectrogram)

        with np.testing.assert_raises(AssertionError):
            np.testing.assert_array_equal(augmented_spectrogram, magnitude_spectrogram)

        num_zeroed_frequencies = 0
        for i in range(augmented_spectrogram.shape[0]):
            if sum(augmented_spectrogram[i]) == 0.0:
                num_zeroed_frequencies += 1

        self.assertEqual(
            num_zeroed_frequencies,
            int(round(magnitude_spectrogram.shape[0] * mask_fraction)),
        )

    def test_fill_mean(self):
        samples, sample_rate = librosa.load(
            os.path.join(DEMO_DIR, "acoustic_guitar_0.wav")
        )
        magnitude_spectrogram = librosa.feature.melspectrogram(
            y=samples, sr=sample_rate
        )

        min_mask_fraction = 0.05
        max_mask_fraction = 0.09
        transform = SpecFrequencyMask(
            fill_mode="mean",
            min_mask_fraction=min_mask_fraction,
            max_mask_fraction=max_mask_fraction,
            p=1.0,
        )
        augmented_spectrogram = transform(magnitude_spectrogram)

        num_masked_frequencies = 0
        for i in range(augmented_spectrogram.shape[0]):
            if (
                np.std(augmented_spectrogram[i]) == 0.0
                and sum(augmented_spectrogram[i]) != 0.0
            ):
                num_masked_frequencies += 1

        self.assertGreaterEqual(
            num_masked_frequencies,
            int(round(magnitude_spectrogram.shape[0] * min_mask_fraction)),
        )
        self.assertLessEqual(
            num_masked_frequencies,
            int(round(magnitude_spectrogram.shape[0] * max_mask_fraction)),
        )
