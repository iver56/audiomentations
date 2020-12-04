import json
import os
import unittest

import numpy as np
from numpy.testing import assert_array_equal

from audiomentations.augmentations.transforms import AddShortNoises
from audiomentations.core.composition import Compose
from audiomentations.core.transforms_interface import (
    MultichannelAudioNotSupportedException,
)
from audiomentations.core.utils import calculate_rms
from demo.demo import DEMO_DIR


class TestAddShortNoises(unittest.TestCase):
    def test_add_short_noises(self):
        sample_rate = 44100
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
        sample_rate = 44100
        samples = np.sin(
            np.linspace(0, 440 * 2 * np.pi, int(0.03125 * sample_rate))
        ).astype(np.float32)
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

    def test_add_silence(self):
        """Check that AddShortNoises does not crash if a noise is completely silent."""
        sample_rate = 48000
        samples = np.sin(np.linspace(0, 440 * 2 * np.pi, 7000)).astype(np.float32)
        augmenter = AddShortNoises(
            sounds_path=os.path.join(DEMO_DIR, "silence"),
            min_time_between_sounds=0.001,
            max_time_between_sounds=0.002,
            p=1.0,
        )

        augmenter.parameters = {
            "should_apply": True,
            "sounds": [
                {
                    "fade_in_time": 0.04257633246838298,
                    "start": -0.00013191289693534575,
                    "end": 0.8071696744046519,
                    "fade_out_time": 0.07119110196424423,
                    "file_path": os.path.join(DEMO_DIR, "silence", "silence.wav"),
                    "snr_in_db": 19.040001423519563,
                }
            ],
        }
        augmenter.freeze_parameters()

        samples_out = augmenter(samples=samples, sample_rate=sample_rate)
        self.assertEqual(samples_out.dtype, np.float32)
        self.assertEqual(samples_out.shape, samples.shape)

    def test_too_long_fade_time(self):
        """Check that a too long fade time does not result in an exception."""
        sample_rate = 44100
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
                    min_fade_in_time=0.9,
                    max_fade_in_time=0.99,
                    min_fade_out_time=0.9,
                    max_fade_out_time=0.99,
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
        transform.randomize_parameters(samples, sample_rate=44100)
        json.dumps(transform.serialize_parameters())

    def test_frozen_parameters(self):
        sample_rate = 44100
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

    def test_multichannel_audio_not_supported_yet(self):
        sample_rate = 44100
        samples_chn0 = np.sin(np.linspace(0, 440 * 2 * np.pi, 2 * sample_rate)).astype(
            np.float32
        )
        samples_chn1 = np.sin(np.linspace(0, 440 * 2 * np.pi, 2 * sample_rate)).astype(
            np.float32
        )
        samples = np.vstack((samples_chn0, samples_chn1))

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
        with self.assertRaises(MultichannelAudioNotSupportedException):
            samples_out = augmenter(samples=samples, sample_rate=sample_rate)
