import math
import os
import unittest
import warnings

import numpy as np

from audiomentations.core.audio_loading_utils import (
    load_sound_file,
)
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

    def test_load_mono_signed_24_bit_wav2(self):
        samples, sample_rate = load_sound_file(
            os.path.join(DEMO_DIR, "mono_int24.wav"), sample_rate=None
        )
        self.assertEqual(sample_rate, 44100)
        self.assertEqual(samples.dtype, np.float32)
        self.assertEqual(samples.ndim, 1)

        self.assertEqual(samples.shape[-1], 22)

        max_value = np.amax(samples)
        self.assertAlmostEqual(max_value, 0.96750367)
        min_value = np.amin(samples)
        self.assertAlmostEqual(min_value, -0.9822748)

    def test_load_mono_signed_32_bit_wav(self):
        samples, sample_rate = load_sound_file(
            os.path.join(DEMO_DIR, "mono_int32.wav"), sample_rate=None
        )
        self.assertEqual(sample_rate, 44100)
        self.assertEqual(samples.dtype, np.float32)
        self.assertEqual(samples.ndim, 1)

        self.assertEqual(samples.shape[-1], 22)

        max_value = np.amax(samples)
        self.assertAlmostEqual(max_value, 0.96750367)
        min_value = np.amin(samples)
        self.assertAlmostEqual(min_value, -0.9822748)

    def test_load_mono_float64_wav(self):
        samples, sample_rate = load_sound_file(
            os.path.join(DEMO_DIR, "mono_float64.wav"), sample_rate=None
        )
        self.assertEqual(sample_rate, 44100)
        self.assertEqual(samples.dtype, np.float32)
        self.assertEqual(samples.ndim, 1)

        self.assertEqual(samples.shape[-1], 22)

        max_value = np.amax(samples)
        self.assertAlmostEqual(max_value, 0.96750367)
        min_value = np.amin(samples)
        self.assertAlmostEqual(min_value, -0.9822748)

    def test_load_stereo_signed_24_bit_wav(self):
        samples, sample_rate = load_sound_file(
            os.path.join(DEMO_DIR, "stereo_24bit.WAV"), sample_rate=None
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
        with warnings.catch_warnings(record=True) as w:
            # Cause all warnings to always be triggered.
            warnings.simplefilter("always")

            samples, sample_rate = load_sound_file(
                os.path.join(DEMO_DIR, "ms_adpcm.wav"), sample_rate=16000
            )

            assert len(w) == 1
            assert (
                "resampled from 11024 hz to 16000 hz. This hurt execution time"
                in str(w[-1].message)
            )

        self.assertEqual(sample_rate, 16000)
        self.assertEqual(samples.dtype, np.float32)
        self.assertEqual(len(samples.shape), 1)

        self.assertEqual(samples.shape[0], math.ceil(895500 * 16000 / 11024))

        max_value = np.amax(samples)
        self.assertGreater(max_value, 0.3)
        self.assertLess(max_value, 1.0)
