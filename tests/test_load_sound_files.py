import math
import os
import unittest

import numpy as np

from audiomentations.core.audio_loading_utils import load_sound_file, load_wav_file_with_wavio
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

    def test_load_mono_m4a(self):
        samples, sample_rate = load_sound_file(
            os.path.join(DEMO_DIR, "testing.m4a"), sample_rate=None
        )
        self.assertEqual(sample_rate, 44100)
        self.assertEqual(samples.dtype, np.float32)
        self.assertEqual(len(samples.shape), 1)

        self.assertGreaterEqual(samples.shape[0], 141312)
        self.assertLessEqual(samples.shape[0], 141312)

        max_value = np.amax(samples)
        self.assertGreater(max_value, 0.1)
        self.assertLess(max_value, 1.0)

    def test_load_mono_signed_16_bit_wav(self):
        samples, sample_rate = load_sound_file(
            os.path.join(DEMO_DIR, "acoustic_guitar_0.wav"), sample_rate=None
        )
        self.assertEqual(sample_rate, 16000)
        self.assertEqual(samples.dtype, np.float32)
        self.assertEqual(len(samples.shape), 1)

        self.assertEqual(samples.shape[0], 140544)

        max_value = np.amax(samples)
        self.assertGreater(max_value, 0.5)
        self.assertLess(max_value, 1.0)

    def test_load_stereo_signed_16_bit_wav(self):
        samples, sample_rate = load_sound_file(
            os.path.join(DEMO_DIR, "stereo_16bit.wav"), sample_rate=None
        )
        self.assertEqual(sample_rate, 16000)
        self.assertEqual(samples.dtype, np.float32)
        self.assertEqual(len(samples.shape), 1)

        self.assertEqual(samples.shape[0], 17833)

        max_value = np.amax(samples)
        self.assertGreater(max_value, 0.5)
        self.assertLess(max_value, 1.0)

    def test_load_mono_signed_16_bit_wav_with_wavio(self):
        samples, sample_rate = load_wav_file_with_wavio(
            os.path.join(DEMO_DIR, "acoustic_guitar_0.wav"), sample_rate=None
        )
        self.assertEqual(sample_rate, 16000)
        self.assertEqual(samples.dtype, np.float32)
        self.assertEqual(len(samples.shape), 1)

        self.assertEqual(samples.shape[0], 140544)

        max_value = np.amax(samples)
        self.assertGreater(max_value, 0.5)
        self.assertLess(max_value, 1.0)

    def test_load_mono_signed_24_bit_wav(self):
        samples, sample_rate = load_sound_file(
            os.path.join(DEMO_DIR, "signed_24bit.wav"), sample_rate=None
        )
        self.assertEqual(sample_rate, 48000)
        self.assertEqual(samples.dtype, np.float32)
        self.assertEqual(len(samples.shape), 1)

        self.assertEqual(samples.shape[0], 54514)

        max_value = np.amax(samples)
        self.assertGreater(max_value, 0.09)
        self.assertLess(max_value, 1.0)

    def test_load_stereo_signed_24_bit_wav(self):
        samples, sample_rate = load_sound_file(
            os.path.join(DEMO_DIR, "stereo_24bit.wav"), sample_rate=None
        )
        self.assertEqual(sample_rate, 16000)
        self.assertEqual(samples.dtype, np.float32)
        self.assertEqual(len(samples.shape), 1)

        self.assertEqual(samples.shape[0], 17833)

        max_value = np.amax(samples)
        self.assertGreater(max_value, 0.5)
        self.assertLess(max_value, 1.0)

    def test_load_mono_ms_adpcm(self):
        samples, sample_rate = load_sound_file(
            os.path.join(DEMO_DIR, "ms_adpcm.wav"), sample_rate=None
        )
        self.assertEqual(sample_rate, 11024)
        self.assertEqual(samples.dtype, np.float32)
        self.assertEqual(len(samples.shape), 1)

        self.assertEqual(samples.shape[0], 895500)

        max_value = np.amax(samples)
        self.assertGreater(max_value, 0.3)
        self.assertLess(max_value, 1.0)

    def test_load_mono_ms_adpcm_and_resample(self):
        samples, sample_rate = load_sound_file(
            os.path.join(DEMO_DIR, "ms_adpcm.wav"), sample_rate=16000
        )
        self.assertEqual(sample_rate, 16000)
        self.assertEqual(samples.dtype, np.float32)
        self.assertEqual(len(samples.shape), 1)

        self.assertEqual(samples.shape[0], math.ceil(895500 * 16000 / 11024))

        max_value = np.amax(samples)
        self.assertGreater(max_value, 0.3)
        self.assertLess(max_value, 1.0)
