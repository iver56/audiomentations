import os
import unittest

import numpy as np

from audiomentations.augmentations.transforms import AddImpulseResponse
from audiomentations.core.composition import Compose
from demo.demo import DEMO_DIR


class TestImpulseResponse(unittest.TestCase):
    def test_dynamic_length(self):
        sample_len = 1024
        samples_in = np.random.normal(0, 1, size=sample_len).astype(np.float32)
        sample_rate = 16000
        augmenter = Compose(
            [AddImpulseResponse(ir_path=os.path.join(DEMO_DIR, "ir"), p=1.0)]
        )

        self.assertEqual(len(samples_in), sample_len)
        samples_out = augmenter(samples=samples_in, sample_rate=sample_rate)

        self.assertEqual(samples_out.dtype, np.float32)
        self.assertGreater(len(samples_out), len(samples_in))
