import os
import random
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import soundfile
from PIL import Image
from librosa.display import specshow
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from audiomentations import AddBackgroundNoise
from audiomentations.core.audio_loading_utils import load_sound_file

transform_usage_example_classes = dict()


def plot_waveforms_and_spectrograms(
    sound, transformed_sound, sample_rate, output_file_path
):
    ylim = max(np.amax(np.abs(sound)), np.amax(np.abs(transformed_sound))) * 1.05

    fig, axs = plt.subplots(2, 2)

    axs[0, 0].plot(sound)
    axs[0, 0].set_xticklabels([])
    axs[0, 0].set_xticks([])
    axs[0, 0].set_ylim([-ylim, ylim])
    axs[0, 0].title.set_text("Input sound")

    axs[0, 1].plot(transformed_sound)
    axs[0, 1].set_xticklabels([])
    axs[0, 1].set_xticks([])
    axs[0, 1].set_ylim([-ylim, ylim])
    axs[0, 1].set_yticks([])
    axs[0, 1].set_yticklabels([])
    axs[0, 1].title.set_text("Transformed sound")

    def get_magnitude_spectrogram(samples):
        complex_spec = librosa.stft(samples)
        return librosa.amplitude_to_db(np.abs(complex_spec), ref=np.max)

    sound_spec = get_magnitude_spectrogram(sound)
    transformed_sound_spec = get_magnitude_spectrogram(transformed_sound)

    vmax = max(np.amax(sound_spec), np.amax(transformed_sound_spec))
    vmin = vmax - 80.0

    specshow(
        sound_spec,
        ax=axs[1, 0],
        vmax=vmax,
        vmin=vmin,
        x_axis="time",
        y_axis="linear",
        sr=sample_rate,
    )
    axs[1, 0].xaxis.set_major_locator(plt.MaxNLocator(5))
    axs[1, 0].set_ylim([0, sample_rate // 2])

    specshow(
        transformed_sound_spec,
        ax=axs[1, 1],
        vmax=vmax,
        vmin=vmin,
        x_axis="time",
        y_axis="linear",
        sr=sample_rate,
    )
    axs[1, 1].xaxis.set_major_locator(plt.MaxNLocator(5))
    axs[1, 1].set_yticks([])
    axs[1, 1].set_yticklabels([])
    axs[1, 1].set_ylabel("")

    plt.savefig(output_file_path, dpi=200)
    plt.close(fig)


class TransformUsageExample:
    transform_class = None

    def generate_example(self) -> Tuple[NDArray, NDArray, int]:
        pass


def register(cls):
    """Register a transform usage example class."""
    transform_usage_example_classes[cls.transform_class] = cls
    return cls


@register
class AddBackgroundNoiseExample(TransformUsageExample):
    transform_class = AddBackgroundNoise

    def generate_example(self):
        random.seed(345)
        np.random.seed(345)
        transform = AddBackgroundNoise(
            sounds_path=librosa.example("pistachio"),
            min_snr_in_db=5.0,
            max_snr_in_db=5.0,
            p=1.0,
        )

        sound, sample_rate = load_sound_file(
            librosa.example("libri1"), sample_rate=16000
        )

        sound = sound[..., 0 : 5 * sample_rate]

        transformed_sound = transform(sound, sample_rate)

        return sound, transformed_sound, sample_rate


if __name__ == "__main__":
    BASE_DIR = Path(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    for transform_class in transform_usage_example_classes:
        transform_usage_example_class = transform_usage_example_classes[transform_class]
        (
            sound,
            transformed_sound,
            sample_rate,
        ) = transform_usage_example_class().generate_example()
        output_file_path = (
            BASE_DIR
            / "docs"
            / "waveform_transforms"
            / f"{transform_class.__name__}.png"
        )
        plot_waveforms_and_spectrograms(
            sound,
            transformed_sound,
            sample_rate,
            output_file_path=output_file_path,
        )
        Image.open(output_file_path).save(
            output_file_path.with_suffix(".webp"), "webp", lossless=True, quality=100
        )
        os.remove(output_file_path)

        soundfile.write(
            BASE_DIR
            / "docs"
            / "waveform_transforms"
            / f"{transform_class.__name__}_input.flac",
            sound,
            sample_rate,
        )
        soundfile.write(
            BASE_DIR
            / "docs"
            / "waveform_transforms"
            / f"{transform_class.__name__}_transformed.flac",
            transformed_sound,
            sample_rate,
        )
