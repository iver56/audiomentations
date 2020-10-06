import os
import unittest

import librosa
import numpy as np

from audiomentations.augmentations.spectrogram_transforms import SpecFrequencyMask
from audiomentations.core.audio_loading_utils import load_sound_file
from demo.demo import DEMO_DIR

DEBUG = False


def plot_matrix(matrix, output_image_path=None, vmin=None, vmax=None, title=None):
    """
    Plot a 2D matrix with viridis color map

    :param matrix: 2D numpy array
    :return:
    """
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    if title is not None:
        ax.set_title(title)
    plt.imshow(matrix, vmin=vmin, vmax=vmax)
    if matrix.shape[-1] != 3:
        plt.colorbar()
    if output_image_path:
        plt.savefig(str(output_image_path), dpi=200)
    else:
        plt.show()
    plt.close(fig)


class TestSpecFrequencyMask(unittest.TestCase):
    def test_fill_zeros(self):
        samples, sample_rate = load_sound_file(
            os.path.join(DEMO_DIR, "acoustic_guitar_0.wav"), sample_rate=None
        )
        magnitude_spectrogram = librosa.feature.melspectrogram(
            y=samples, sr=sample_rate
        )

        mask_fraction = 0.05
        transform = SpecFrequencyMask(
            fill_mode="constant",
            fill_constant=0.0,
            min_mask_fraction=mask_fraction,
            max_mask_fraction=mask_fraction,
            p=1.0,
        )
        augmented_spectrogram = transform(magnitude_spectrogram)

        if DEBUG:
            plot_matrix(np.log(augmented_spectrogram))

        with np.testing.assert_raises(AssertionError):
            np.testing.assert_array_equal(augmented_spectrogram, magnitude_spectrogram)

        num_zeroed_frequencies = 0
        for i in range(augmented_spectrogram.shape[0]):
            if sum(augmented_spectrogram[i]) == 0.0:
                num_zeroed_frequencies += 1

        self.assertEqual(
            num_zeroed_frequencies,
            int(round(magnitude_spectrogram.shape[0] * mask_fraction)),
        )

    def test_fill_zeros_multichannel(self):
        samples, sample_rate = load_sound_file(
            os.path.join(DEMO_DIR, "background_noises", "hens.ogg"),
            sample_rate=None,
            mono=False,
        )
        assert samples.shape[0] == 2
        magnitude_spectrogram_chn0 = librosa.feature.melspectrogram(
            y=samples[0, :], sr=sample_rate
        )
        magnitude_spectrogram_chn1 = librosa.feature.melspectrogram(
            y=samples[1, :], sr=sample_rate
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
        multichannel_magnitude_spectrogram[:, :, 2] = magnitude_spectrogram_chn1

        mask_fraction = 0.05
        transform = SpecFrequencyMask(
            fill_mode="constant",
            fill_constant=0.0,
            min_mask_fraction=mask_fraction,
            max_mask_fraction=mask_fraction,
            p=1.0,
        )
        augmented_spectrogram = transform(multichannel_magnitude_spectrogram)

        if DEBUG:
            image = (7 + np.log10(augmented_spectrogram + 0.0000001)) / 8
            plot_matrix(image)

        with np.testing.assert_raises(AssertionError):
            np.testing.assert_array_equal(
                augmented_spectrogram, multichannel_magnitude_spectrogram
            )

        num_zeroed_frequencies = 0
        for i in range(augmented_spectrogram.shape[0]):
            if np.sum(augmented_spectrogram[i]) == 0.0:
                num_zeroed_frequencies += 1

        self.assertEqual(
            num_zeroed_frequencies,
            int(round(multichannel_magnitude_spectrogram.shape[0] * mask_fraction)),
        )

    def test_fill_mean(self):
        samples, sample_rate = load_sound_file(
            os.path.join(DEMO_DIR, "acoustic_guitar_0.wav"), sample_rate=None
        )
        magnitude_spectrogram = librosa.feature.melspectrogram(
            y=samples, sr=sample_rate
        )

        min_mask_fraction = 0.05
        max_mask_fraction = 0.09
        transform = SpecFrequencyMask(
            fill_mode="mean",
            min_mask_fraction=min_mask_fraction,
            max_mask_fraction=max_mask_fraction,
            p=1.0,
        )
        augmented_spectrogram = transform(magnitude_spectrogram)

        if DEBUG:
            plot_matrix(np.log(augmented_spectrogram))

        num_masked_frequencies = 0
        for i in range(augmented_spectrogram.shape[0]):
            frequency_slice = augmented_spectrogram[i]
            if (
                np.amin(frequency_slice) == np.amax(frequency_slice)
                and sum(frequency_slice) != 0.0
            ):
                num_masked_frequencies += 1

        self.assertGreaterEqual(
            num_masked_frequencies,
            int(round(magnitude_spectrogram.shape[0] * min_mask_fraction)),
        )
        self.assertLessEqual(
            num_masked_frequencies,
            int(round(magnitude_spectrogram.shape[0] * max_mask_fraction)),
        )

    def test_fill_mean_multichannel(self):
        samples, sample_rate = load_sound_file(
            os.path.join(DEMO_DIR, "background_noises", "hens.ogg"),
            sample_rate=None,
            mono=False,
        )
        assert samples.shape[0] == 2
        magnitude_spectrogram_chn0 = librosa.feature.melspectrogram(
            y=samples[0, :], sr=sample_rate
        )
        magnitude_spectrogram_chn1 = librosa.feature.melspectrogram(
            y=samples[1, :], sr=sample_rate
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
        multichannel_magnitude_spectrogram[:, :, 2] = magnitude_spectrogram_chn1

        mask_fraction = 0.05
        transform = SpecFrequencyMask(
            fill_mode="mean",
            min_mask_fraction=mask_fraction,
            max_mask_fraction=mask_fraction,
            p=1.0,
        )
        augmented_spectrogram = transform(multichannel_magnitude_spectrogram)

        if DEBUG:
            image = (7 + np.log10(augmented_spectrogram + 0.0000001)) / 8
            plot_matrix(image)

        with np.testing.assert_raises(AssertionError):
            np.testing.assert_array_equal(
                augmented_spectrogram, multichannel_magnitude_spectrogram
            )

        num_masked_frequencies = 0
        for i in range(augmented_spectrogram.shape[0]):
            frequency_slice = augmented_spectrogram[i]
            if (
                np.amin(frequency_slice) == np.amax(frequency_slice)
                and np.sum(frequency_slice) != 0.0
            ):
                num_masked_frequencies += 1

        self.assertEqual(
            num_masked_frequencies,
            int(round(multichannel_magnitude_spectrogram.shape[0] * mask_fraction)),
        )
