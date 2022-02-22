import unittest

import numpy as np
from numpy.testing import assert_array_equal

from audiomentations import Padding, Compose


class TestPadding(unittest.TestCase):
    def test_padding_constant(self):
        samples = np.array([0.5, 0.6, -0.2, 0.0], dtype=np.float32)
        sample_rate = 16000
        orig_len = len(samples)
        augmenter = Compose([Padding(mode='constant', p=1.0)])
        samples = augmenter(samples=samples, sample_rate=sample_rate)

        self.assertEqual(samples.dtype, np.float32)
        self.assertEqual(len(samples), orig_len)

    def test_padding_edge(self):
        samples = np.array([0.5, 0.6, -0.8, 0.0], dtype=np.float32)
        sample_rate = 16000
        orig_len = len(samples)
        augmenter = Compose([Padding(mode='edge', p=1.0)])
        samples = augmenter(samples=samples, sample_rate=sample_rate)

        self.assertEqual(samples.dtype, np.float32)
        self.assertEqual(len(samples), orig_len)

    def test_padding_wrap(self):
        samples = np.array([0.5, 0.6, -0.8, 0.0], dtype=np.float32)
        sample_rate = 16000
        orig_len = len(samples)
        augmenter = Compose([Padding(mode='wrap', p=1.0)])
        samples = augmenter(samples=samples, sample_rate=sample_rate)

        self.assertEqual(samples.dtype, np.float32)
        self.assertEqual(len(samples), orig_len)
        
    def test_padding_reflect(self):
        samples = np.array([0.5, 0.6, -0.8, 0.0], dtype=np.float32)
        sample_rate = 16000
        orig_len = len(samples)
        augmenter = Compose([Padding(mode='reflect', p=1.0)])
        samples = augmenter(samples=samples, sample_rate=sample_rate)

        self.assertEqual(samples.dtype, np.float32)
        self.assertEqual(len(samples), orig_len)

    def test_padding_constant_multichannel(self):
        samples = np.array(
            [[0.9, 0.5, -0.25, -0.125, 0.0], [0.95, 0.5, -0.25, -0.125, 0.0]],
            dtype=np.float32,
        )
        sample_rate = 16000

        augmenter = Compose([Padding(mode='constant', p=1.0)])
        samples = augmenter(samples=samples, sample_rate=sample_rate)

        self.assertEqual(samples.dtype, np.float32)
