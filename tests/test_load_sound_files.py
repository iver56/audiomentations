import os
import unittest

import numpy as np

from audiomentations.core.utils import load_sound_file
from demo.demo import DEMO_DIR


class TestLoadSoundFiles(unittest.TestCase):
    def test_load_stereo_ogg_vorbis(self):
        samples, sample_rate = load_sound_file(
            os.path.join(DEMO_DIR, "background_noises", "hens.ogg"), sample_rate=None
        )
        self.assertEqual(samples.dtype, np.float32)
        self.assertEqual(len(samples.shape), 1)

        # Apparently, the exact duration may vary slightly based on which decoder is used
        self.assertGreaterEqual(samples.shape[0], 442575)
        self.assertLessEqual(samples.shape[0], 443328)

        max_value = np.amax(samples)
        self.assertGreater(max_value, 0.02)
        self.assertLess(max_value, 1.0)

    def test_load_mono_opus(self):
        samples, sample_rate = load_sound_file(
            os.path.join(DEMO_DIR, "bus.opus"), sample_rate=None
        )
        self.assertEqual(samples.dtype, np.float32)
        self.assertEqual(len(samples.shape), 1)

        # Apparently, the exact duration may vary slightly based on which decoder is used
        self.assertGreaterEqual(samples.shape[0], 36682)
        self.assertLessEqual(samples.shape[0], 36994)

        max_value = np.amax(samples)
        self.assertGreater(max_value, 0.3)
        self.assertLess(max_value, 1.0)
