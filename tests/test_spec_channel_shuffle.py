import os
import unittest

import librosa
import numpy as np

from audiomentations import SpecChannelShuffle
from audiomentations.core.audio_loading_utils import load_sound_file
from audiomentations.core.transforms_interface import MonoAudioNotSupportedException
from demo.demo import DEMO_DIR
from .utils import plot_matrix

DEBUG = False


class TestSpecChannelShuffle(unittest.TestCase):
    def test_shuffle_channels(self):
        samples, sample_rate = load_sound_file(
            os.path.join(DEMO_DIR, "background_noises", "hens.ogg"),
            sample_rate=None,
            mono=False,
        )
        assert samples.shape[0] == 2
        magnitude_spectrogram_chn0 = librosa.feature.melspectrogram(
            y=np.asfortranarray(samples[0, :]), sr=sample_rate
        )
        magnitude_spectrogram_chn1 = librosa.feature.melspectrogram(
            y=np.asfortranarray(samples[1, :]), sr=sample_rate
        )
        multichannel_magnitude_spectrogram = np.zeros(
            shape=(
                magnitude_spectrogram_chn0.shape[0],
                magnitude_spectrogram_chn0.shape[1],
                3,
            ),
            dtype=np.float32,
        )
        multichannel_magnitude_spectrogram[:, :, 0] = magnitude_spectrogram_chn0
        multichannel_magnitude_spectrogram[:, :, 1] = magnitude_spectrogram_chn1
        multichannel_magnitude_spectrogram[:, :, 2] = magnitude_spectrogram_chn1 * 0.7

        if DEBUG:
            image = (7 + np.log10(multichannel_magnitude_spectrogram + 0.0000001)) / 8
            plot_matrix(image, title="before")

        # Make the shuffled channels do not equal the original order
        transform = SpecChannelShuffle(p=1.0)
        for _ in range(100000):
            transform.randomize_parameters(multichannel_magnitude_spectrogram)
            if transform.parameters["shuffled_channel_indexes"] != [0, 1, 2]:
                break
        transform.freeze_parameters()

        augmented_spectrogram = transform(multichannel_magnitude_spectrogram)

        if DEBUG:
            image = (7 + np.log10(augmented_spectrogram + 0.0000001)) / 8
            plot_matrix(image, title="after")

        with np.testing.assert_raises(AssertionError):
            np.testing.assert_array_equal(
                augmented_spectrogram, multichannel_magnitude_spectrogram
            )

        for augmented_index, original_index in enumerate(
            transform.parameters.get("shuffled_channel_indexes")
        ):
            np.testing.assert_array_equal(
                augmented_spectrogram[:, :, augmented_index],
                multichannel_magnitude_spectrogram[:, :, original_index],
            )

    def test_shuffle_channels_mono(self):
        samples, sample_rate = load_sound_file(
            os.path.join(DEMO_DIR, "acoustic_guitar_0.wav"), sample_rate=None
        )
        magnitude_spectrogram = librosa.feature.melspectrogram(
            y=samples, sr=sample_rate
        )

        transform = SpecChannelShuffle(p=1.0)
        with self.assertRaises(MonoAudioNotSupportedException):
            augmented_spectrogram = transform(magnitude_spectrogram)

    def test_empty_spectrogram(self):
        spec = np.zeros(shape=(0, 0), dtype=np.float32)
        transform = SpecChannelShuffle(p=1.0)
        augmented_spectrogram = transform(spec)

        np.testing.assert_array_equal(spec, augmented_spectrogram)
