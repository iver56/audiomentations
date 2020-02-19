import json
import os
import unittest

import numpy as np
from numpy.testing import assert_array_equal

from audiomentations import calculate_rms
from audiomentations.augmentations.transforms import AddShortNoises
from audiomentations.core.composition import Compose
from demo.demo import DEMO_DIR


class TestAddShortNoises(unittest.TestCase):
    def test_add_short_noises(self):
        sample_rate = 16000
        samples = np.sin(np.linspace(0, 440 * 2 * np.pi, 9 * sample_rate)).astype(
            np.float32
        )
        rms_before = calculate_rms(samples)
        augmenter = Compose(
            [
                AddShortNoises(
                    sounds_path=os.path.join(DEMO_DIR, "short_noises"),
                    min_time_between_sounds=2.0,
                    max_time_between_sounds=8.0,
                    p=1.0,
                )
            ]
        )
        samples_out = augmenter(samples=samples, sample_rate=sample_rate)
        self.assertEqual(samples_out.dtype, np.float32)
        self.assertEqual(samples_out.shape, samples.shape)

        rms_after = calculate_rms(samples_out)
        self.assertGreater(rms_after, rms_before)

    def test_input_shorter_than_noise(self):
        """
        Verify correct behavior when the input sound is shorter than the added noise sounds.
        """
        sample_rate = 16000
        samples = np.sin(np.linspace(0, 440 * 2 * np.pi, 500)).astype(
            np.float32
        )
        rms_before = calculate_rms(samples)
        augmenter = Compose(
            [
                AddShortNoises(
                    sounds_path=os.path.join(DEMO_DIR, "short_noises"),
                    min_time_between_sounds=0.00001,
                    max_time_between_sounds=0.00002,
                    p=1.0,
                )
            ]
        )
        samples_out = augmenter(samples=samples, sample_rate=sample_rate)
        self.assertEqual(samples_out.dtype, np.float32)
        self.assertEqual(samples_out.shape, samples.shape)

        rms_after = calculate_rms(samples_out)
        self.assertGreater(rms_after, rms_before)

    def test_serialize_parameters(self):
        transform = AddShortNoises(
            sounds_path=os.path.join(DEMO_DIR, "background_noises"), p=1.0
        )
        samples = np.random.normal(0, 1, size=1024).astype(np.float32)
        transform.randomize_parameters(samples, sample_rate=16000)
        json.dumps(transform.serialize_parameters())

    def test_frozen_parameters(self):
        sample_rate = 16000
        samples = np.sin(np.linspace(0, 440 * 2 * np.pi, 9 * sample_rate)).astype(
            np.float32
        )
        augmenter = Compose(
            [
                AddShortNoises(
                    sounds_path=os.path.join(DEMO_DIR, "short_noises"),
                    min_time_between_sounds=2.0,
                    max_time_between_sounds=8.0,
                    p=1.0,
                )
            ]
        )
        samples_out1 = augmenter(samples=samples, sample_rate=sample_rate)

        augmenter.freeze_parameters()
        samples_out2 = augmenter(samples, sample_rate)

        assert_array_equal(samples_out1, samples_out2)
