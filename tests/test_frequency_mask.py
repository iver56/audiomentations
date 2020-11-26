import unittest

import numpy as np
from numpy.testing import assert_array_equal

from audiomentations.augmentations.transforms import FrequencyMask
from audiomentations.core.composition import Compose


class TestFrequencyMask(unittest.TestCase):
    def test_apply_frequency_mask(self):
        sample_len = 1024
        samples_in = np.random.normal(0, 1, size=sample_len).astype(np.float32)
        samples_in_copy = np.copy(samples_in)
        sample_rate = 16000
        augmenter = Compose(
            [FrequencyMask(min_frequency_band=0.3, max_frequency_band=0.5, p=1.0)]
        )

        samples_out = augmenter(samples=samples_in, sample_rate=sample_rate)
        self.assertEqual(samples_out.dtype, np.float32)
        self.assertEqual(len(samples_out), sample_len)

        # Check that the input is left untouched
        assert_array_equal(samples_in, samples_in_copy)

        std_in = np.mean(np.abs(samples_in))
        std_out = np.mean(np.abs(samples_out))
        self.assertLess(std_out, std_in)

    def test_apply_frequency_mask_stereo(self):
        np.random.seed(42)
        sample_len = 1024
        samples_in = np.random.uniform(low=-0.5, high=0.5, size=(2, sample_len)).astype(
            np.float32
        )
        samples_in_copy = np.copy(samples_in)
        sample_rate = 16000
        augmenter = FrequencyMask(min_frequency_band=0.3, max_frequency_band=0.3, p=1.0)

        samples_out = augmenter(samples=samples_in, sample_rate=sample_rate)
        self.assertEqual(samples_out.dtype, np.float32)
        self.assertEqual(samples_out.shape, samples_in.shape)

        # Check that the input is left untouched
        assert_array_equal(samples_in, samples_in_copy)

        std_in = np.mean(np.abs(samples_in))
        std_out = np.mean(np.abs(samples_out))
        self.assertLess(std_out, std_in)

        augmenter.freeze_parameters()
        samples_only_1st_channel = augmenter(
            samples=samples_in[0, :], sample_rate=sample_rate
        )

        assert_array_equal(samples_out[0, :], samples_only_1st_channel)

    def test_filter_instability(self):
        """
        An early implementation of FrequencyMask had a problem with filter instability
        sometimes. That would lead to NaN values in the result. This test checks whether or not
        the problem currently exists.
        """
        np.random.seed(42)
        sample_len = 32000
        samples_in = np.random.uniform(-1, 1, size=sample_len).astype(np.float32)

        sample_rate = 16000
        augmenter = Compose([FrequencyMask(p=1.0)])

        augmenter.transforms[0].randomize_parameters(samples_in, sample_rate)
        augmenter.transforms[0].parameters["bandwidth"] = 600
        augmenter.transforms[0].parameters["freq_start"] = 17
        augmenter.freeze_parameters()

        samples_out = augmenter(samples=samples_in, sample_rate=sample_rate)
        self.assertFalse(np.isnan(samples_out).any())
