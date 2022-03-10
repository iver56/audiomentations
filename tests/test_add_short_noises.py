import json
import os
import pickle
import random

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal

from audiomentations import AddShortNoises, PolarityInversion
from audiomentations.core.transforms_interface import (
    MultichannelAudioNotSupportedException,
)
from audiomentations.core.utils import calculate_rms
from demo.demo import DEMO_DIR


class TestAddShortNoises:
    def test_add_short_noises(self):
        sample_rate = 44100
        samples = np.sin(np.linspace(0, 440 * 2 * np.pi, 9 * sample_rate)).astype(
            np.float32
        )
        rms_before = calculate_rms(samples)
        augmenter = AddShortNoises(
            sounds_path=os.path.join(DEMO_DIR, "short_noises"),
            min_time_between_sounds=2.0,
            max_time_between_sounds=8.0,
            p=1.0,
        )
        samples_out = augmenter(samples=samples, sample_rate=sample_rate)
        assert samples_out.dtype == np.float32
        assert samples_out.shape == samples.shape

        rms_after = calculate_rms(samples_out)
        assert rms_after > rms_before

    def test_add_short_noises_with_signal_gain_during_noise(self):
        sample_rate = 44100
        samples = np.sin(np.linspace(0, 440 * 2 * np.pi, 9 * sample_rate)).astype(
            np.float32
        )
        rms_before = calculate_rms(samples)
        augmenter = AddShortNoises(
            sounds_path=os.path.join(DEMO_DIR, "short_noises"),
            min_snr_in_db=50.0,
            max_snr_in_db=50.0,
            min_time_between_sounds=2.0,
            max_time_between_sounds=4.0,
            signal_gain_in_db_during_noise=-100,
            p=1.0,
        )
        samples_out = augmenter(samples=samples, sample_rate=sample_rate)
        assert samples_out.dtype == np.float32
        assert samples_out.shape == samples.shape

        rms_after = calculate_rms(samples_out)
        assert rms_after < rms_before

    def test_add_short_noises_with_noise_transform(self):
        sample_rate = 44100
        samples = np.sin(np.linspace(0, 440 * 2 * np.pi, 9 * sample_rate)).astype(
            np.float32
        )
        augmenter = AddShortNoises(
            sounds_path=os.path.join(DEMO_DIR, "short_noises"),
            min_snr_in_db=50.0,
            max_snr_in_db=50.0,
            min_time_between_sounds=1.0,
            max_time_between_sounds=2.0,
            noise_transform=PolarityInversion(p=1.0),
            p=1.0,
        )
        samples_out_with_noise_transform = augmenter(
            samples=samples, sample_rate=sample_rate
        )
        assert samples_out_with_noise_transform.dtype == np.float32
        assert samples_out_with_noise_transform.shape == samples.shape

        augmenter.noise_transform = None
        samples_out_without_noise_transform = augmenter(
            samples=samples, sample_rate=sample_rate
        )

        with np.testing.assert_raises(AssertionError):
            assert_array_almost_equal(
                samples_out_without_noise_transform, samples_out_with_noise_transform
            )

    def test_input_shorter_than_noise(self):
        """
        Verify correct behavior when the input sound is shorter than the added noise sounds.
        """
        sample_rate = 44100
        samples = np.sin(
            np.linspace(0, 440 * 2 * np.pi, int(0.03125 * sample_rate))
        ).astype(np.float32)
        rms_before = calculate_rms(samples)
        augmenter = AddShortNoises(
            sounds_path=os.path.join(DEMO_DIR, "short_noises"),
            min_time_between_sounds=0.00001,
            max_time_between_sounds=0.00002,
            p=1.0,
        )
        samples_out = augmenter(samples=samples, sample_rate=sample_rate)
        assert samples_out.dtype == np.float32
        assert samples_out.shape == samples.shape

        rms_after = calculate_rms(samples_out)
        assert rms_after > rms_before

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
        assert samples_out.dtype == np.float32
        assert samples_out.shape == samples.shape

    def test_too_long_fade_time(self):
        """Check that a too long fade time does not result in an exception."""
        sample_rate = 44100
        samples = np.sin(np.linspace(0, 440 * 2 * np.pi, 9 * sample_rate)).astype(
            np.float32
        )
        rms_before = calculate_rms(samples)
        augmenter = AddShortNoises(
            sounds_path=os.path.join(DEMO_DIR, "short_noises"),
            min_time_between_sounds=2.0,
            max_time_between_sounds=8.0,
            min_fade_in_time=0.9,
            max_fade_in_time=0.99,
            min_fade_out_time=0.9,
            max_fade_out_time=0.99,
            p=1.0,
        )
        samples_out = augmenter(samples=samples, sample_rate=sample_rate)
        assert samples_out.dtype == np.float32
        assert samples_out.shape == samples.shape

        rms_after = calculate_rms(samples_out)
        assert rms_after > rms_before

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
        augmenter = AddShortNoises(
            sounds_path=os.path.join(DEMO_DIR, "short_noises"),
            min_time_between_sounds=2.0,
            max_time_between_sounds=8.0,
            p=1.0,
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

        augmenter = AddShortNoises(
            sounds_path=os.path.join(DEMO_DIR, "short_noises"),
            min_time_between_sounds=2.0,
            max_time_between_sounds=8.0,
            p=1.0,
        )
        with pytest.raises(MultichannelAudioNotSupportedException):
            augmenter(samples=samples, sample_rate=sample_rate)

    def test_picklability(self):
        transform = AddShortNoises(
            sounds_path=os.path.join(DEMO_DIR, "short_noises"), p=1.0
        )
        pickled = pickle.dumps(transform)
        unpickled = pickle.loads(pickled)
        assert transform.sound_file_paths == unpickled.sound_file_paths

    def test_noise_rms_parameter(self):
        np.random.seed(80085)
        random.seed("ling")
        sample_rate = 44100
        samples = np.sin(np.linspace(0, 440 * 2 * np.pi, 9 * sample_rate)).astype(
            np.float32
        )
        rms_before = calculate_rms(samples)
        augmenter_relative_to_whole_input = AddShortNoises(
            sounds_path=os.path.join(DEMO_DIR, "short_noises"),
            min_time_between_sounds=2.0,
            max_time_between_sounds=8.0,
            min_snr_in_db=5,
            max_snr_in_db=5,
            noise_rms="relative_to_whole_input",
            p=1.0,
        )

        augmenter_absolute = AddShortNoises(
            sounds_path=os.path.join(DEMO_DIR, "short_noises"),
            min_time_between_sounds=2.0,
            max_time_between_sounds=8.0,
            noise_rms="absolute",
            p=1.0,
        )

        samples_out_absolute = augmenter_absolute(
            samples=samples, sample_rate=sample_rate
        )
        samples_out_relative_to_whole_input = augmenter_relative_to_whole_input(
            samples=samples, sample_rate=sample_rate
        )
        assert samples_out_absolute.dtype == np.float32
        assert samples_out_absolute.shape == samples.shape
        rms_after_relative_to_whole_path = calculate_rms(
            samples_out_relative_to_whole_input
        )
        rms_after_absolute = calculate_rms(samples_out_absolute)
        assert rms_after_absolute > rms_before
        assert rms_after_relative_to_whole_path > rms_before

    def test_include_silence_in_noise_rms_calculation(self):
        np.random.seed(420)
        random.seed(420)
        sample_rate = 44100
        samples = np.sin(np.linspace(0, 440 * 2 * np.pi, 9 * sample_rate)).astype(
            np.float32
        )
        rms_before = calculate_rms(samples)
        augmenter = AddShortNoises(
            sounds_path=os.path.join(DEMO_DIR, "short_noises"),
            min_time_between_sounds=2.0,
            max_time_between_sounds=4.0,
            noise_rms="absolute",
            include_silence_in_noise_rms_estimation=False,
            p=1.0,
        )

        samples_out = augmenter(samples=samples, sample_rate=sample_rate)

        rms_after = calculate_rms(samples_out)
        assert rms_after > rms_before

    def test_add_noises_with_same_level(self):
        dummy_audio = np.random.randint(1, 5, 250000)
        transform_same_noise_level = AddShortNoises(
            sounds_path=os.path.join(DEMO_DIR, "short_noises"),
            min_snr_in_db=15,
            max_snr_in_db=30,
            noise_rms="relative",
            add_all_noises_with_same_level=True,
            min_time_between_sounds=0.5,
            max_time_between_sounds=1,
            p=1,
        )

        transform_different_noise_level = AddShortNoises(
            sounds_path=os.path.join(DEMO_DIR, "short_noises"),
            min_snr_in_db=15,
            max_snr_in_db=30,
            add_all_noises_with_same_level=False,
            min_time_between_sounds=0.5,
            max_time_between_sounds=1,
            p=1,
        )

        for i in range(3):
            transform_same_noise_level.randomize_parameters(
                dummy_audio, sample_rate=44100
            )
            sounds = transform_same_noise_level.parameters["sounds"]
            snr_sounds_same_level = [sounds[j]["snr_in_db"] for j in range(len(sounds))]
            transform_different_noise_level.randomize_parameters(
                dummy_audio, sample_rate=44100
            )
            sounds = transform_different_noise_level.parameters["sounds"]
            snr_sounds_different_level = [
                sounds[j]["snr_in_db"] for j in range(len(sounds))
            ]

            assert len(set(snr_sounds_same_level)) == 1
            assert len(set(snr_sounds_different_level)) > 1
