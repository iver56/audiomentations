import os
import unittest

import numpy as np

from audiomentations.augmentations.transforms import AddImpulseResponse
from audiomentations.core.composition import Compose
from demo.demo import DEMO_DIR


class TestImpulseResponse(unittest.TestCase):
    def test_apply_impulse_response(self):
        sample_len = 1024
        samples_in = np.random.normal(0, 1, size=sample_len).astype(np.float32)
        sample_rate = 16000

        add_ir_transform = AddImpulseResponse(
            ir_path=os.path.join(DEMO_DIR, "ir"), p=1.0
        )

        # Check that misc_file.txt is not one of the IR file candidates, as it's not audio
        self.assertEqual(len(add_ir_transform.ir_files), 1)

        augmenter = Compose([add_ir_transform])

        self.assertEqual(len(samples_in), sample_len)
        samples_out = augmenter(samples=samples_in, sample_rate=sample_rate)

        # Check parameters
        self.assertTrue(augmenter.transforms[0].parameters["should_apply"])
        self.assertEqual(
            augmenter.transforms[0].parameters["ir_file_path"],
            os.path.join(DEMO_DIR, "ir", "impulse_response_0.wav"),
        )

        self.assertEqual(samples_out.dtype, np.float32)
        self.assertGreater(len(samples_out), len(samples_in))
