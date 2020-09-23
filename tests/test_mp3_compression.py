import unittest

import numpy as np

from audiomentations.augmentations.transforms import Mp3Compression
from audiomentations.core.composition import Compose


class TestMp3Compression(unittest.TestCase):
    def test_apply_mp3_compression_pydub(self):
        sample_len = 44100
        samples_in = np.random.normal(0, 1, size=sample_len).astype(np.float32)
        sample_rate = 44100
        augmenter = Compose(
            [Mp3Compression(p=1.0, min_bitrate=48, max_bitrate=48, backend="pydub")]
        )

        samples_out = augmenter(samples=samples_in, sample_rate=sample_rate)
        self.assertEqual(samples_out.dtype, np.float32)
        self.assertGreaterEqual(len(samples_out), sample_len)
        self.assertLess(len(samples_out), sample_len + 2500)

    def test_apply_mp3_compression_lameenc(self):
        sample_len = 44100
        samples_in = np.random.normal(0, 1, size=sample_len).astype(np.float32)
        sample_rate = 44100
        augmenter = Compose(
            [Mp3Compression(p=1.0, min_bitrate=48, max_bitrate=48, backend="lameenc")]
        )

        samples_out = augmenter(samples=samples_in, sample_rate=sample_rate)
        self.assertEqual(samples_out.dtype, np.float32)
        self.assertGreaterEqual(len(samples_out), sample_len)
        self.assertLess(len(samples_out), sample_len + 2500)

    def test_apply_mp3_compression_low_bitrate_pydub(self):
        sample_len = 16000
        samples_in = np.random.normal(0, 1, size=sample_len).astype(np.float32)
        sample_rate = 16000
        augmenter = Compose(
            [Mp3Compression(p=1.0, min_bitrate=8, max_bitrate=8, backend="pydub")]
        )

        samples_out = augmenter(samples=samples_in, sample_rate=sample_rate)
        self.assertEqual(samples_out.dtype, np.float32)
        self.assertGreaterEqual(len(samples_out), sample_len)
        self.assertLess(len(samples_out), sample_len + 2500)

    def test_apply_mp3_compression_low_bitrate_lameenc(self):
        sample_len = 16000
        samples_in = np.random.normal(0, 1, size=sample_len).astype(np.float32)
        sample_rate = 16000
        augmenter = Compose(
            [Mp3Compression(p=1.0, min_bitrate=8, max_bitrate=8, backend="lameenc")]
        )

        samples_out = augmenter(samples=samples_in, sample_rate=sample_rate)
        self.assertEqual(samples_out.dtype, np.float32)
        self.assertGreaterEqual(len(samples_out), sample_len)
        self.assertLess(len(samples_out), sample_len + 2500)
