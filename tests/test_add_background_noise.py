import json
import os
import unittest

import numpy as np

from audiomentations.augmentations.transforms import AddBackgroundNoise
from audiomentations.core.composition import Compose
from demo.demo import DEMO_DIR


class TestAddBackgroundNoise(unittest.TestCase):
    def test_add_background_noise(self):
        samples = np.sin(np.linspace(0, 440 * 2 * np.pi, 8000)).astype(np.float32)
        sample_rate = 16000
        augmenter = Compose(
            [
                AddBackgroundNoise(
                    sounds_path=os.path.join(DEMO_DIR, "background_noises"),
                    min_snr_in_db=15,
                    max_snr_in_db=35,
                    p=1.0,
                )
            ]
        )
        samples_out = augmenter(samples=samples, sample_rate=sample_rate)
        self.assertEqual(samples_out.dtype, np.float32)

    def test_add_background_noise_when_noise_sound_is_too_short(self):
        samples = np.sin(np.linspace(0, 440 * 2 * np.pi, 224000)).astype(np.float32)
        sample_rate = 16000
        augmenter = Compose(
            [
                AddBackgroundNoise(
                    sounds_path=os.path.join(DEMO_DIR, "background_noises"),
                    min_snr_in_db=15,
                    max_snr_in_db=35,
                    p=1.0,
                )
            ]
        )
        samples_out = augmenter(samples=samples, sample_rate=sample_rate)
        self.assertEqual(samples_out.dtype, np.float32)

    def test_serialize_parameters(self):
        transform = AddBackgroundNoise(
            sounds_path=os.path.join(DEMO_DIR, "background_noises"), p=1.0
        )
        samples = np.random.normal(0, 1, size=1024).astype(np.float32)
        transform.randomize_parameters(samples, sample_rate=16000)
        json.dumps(transform.serialize_parameters())
