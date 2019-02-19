import unittest

import numpy as np

from audiomentations.augmentations.transforms import PitchShift
from audiomentations.core.composition import Compose


class TestPitchShift(unittest.TestCase):
    def test_dynamic_length(self):
        samples = np.zeros((512,), dtype=np.float32)
        sample_rate = 16000
        augmenter = Compose([
            PitchShift(min_semitones=-2, max_semitones=-1, p=1.0)
        ])
        samples = augmenter(samples=samples, sample_rate=sample_rate)

        self.assertEqual(samples.dtype, np.float32)
        self.assertEqual(len(samples), 512)
