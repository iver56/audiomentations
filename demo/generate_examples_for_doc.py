import argparse
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

from audiomentations import (
    AddBackgroundNoise,
    AddGaussianNoise,
    AddShortNoises,
    AirAbsorption,
    ApplyImpulseResponse,
    BandPassFilter,
    BandStopFilter,
    Clip,
    ClippingDistortion,
    Gain,
    GainTransition,
    HighPassFilter,
    HighShelfFilter,
    LoudnessNormalization,
    LowPassFilter,
    LowShelfFilter,
    Mp3Compression,
    Normalize,
    Padding,
    PeakingFilter,
    PitchShift,
    PolarityInversion,
    RepeatPart,
    Resample,
    Reverse,
    RoomSimulator,
    SevenBandParametricEQ,
    Shift,
    TanhDistortion,
    TimeMask,
    TimeStretch,
    Trim,
    AdjustDuration,
    Limiter,
    AddGaussianSNR,
    BitCrush,
)
from audiomentations.core.audio_loading_utils import load_sound_file
from audiomentations.core.utils import get_max_abs_amplitude

DEMO_DIR = os.path.dirname(__file__)
transform_usage_example_classes = dict()


def plot_waveforms_and_spectrograms(
    sound, transformed_sound, sample_rate, output_file_path
):
    xmax = max(sound.shape[0], transformed_sound.shape[0])
    ylim = (
        max(get_max_abs_amplitude(sound), get_max_abs_amplitude(transformed_sound))
        * 1.1
    )
    sound = sound[:xmax]
    transformed_sound = transformed_sound[:xmax]
    fig, axs = plt.subplots(
        2,
        2,
        gridspec_kw=dict(
            width_ratios=[sound.shape[-1], transformed_sound.shape[-1]],
            height_ratios=[1, 1],
        ),
    )

    axs[0, 0].plot(sound)
    axs[0, 0].set_xticklabels([])
    axs[0, 0].set_xticks([])
    axs[0, 0].set_xlim([0, sound.shape[-1]])
    axs[0, 0].set_ylim([-ylim, ylim])
    axs[0, 0].title.set_text("Input sound")

    axs[0, 1].plot(transformed_sound)
    axs[0, 1].set_xticklabels([])
    axs[0, 1].set_xticks([])
    axs[0, 1].set_xlim([0, transformed_sound.shape[-1]])
    axs[0, 1].set_ylim([-ylim, ylim])
    axs[0, 1].set_yticks([])
    axs[0, 1].set_yticklabels([])
    axs[0, 1].title.set_text("Transformed sound")

    def get_magnitude_spectrogram(samples):
        complex_spec = librosa.stft(samples)
        return np.abs(complex_spec)

    sound_spec = get_magnitude_spectrogram(sound)
    transformed_sound_spec = get_magnitude_spectrogram(transformed_sound)

    vmax = max(np.amax(sound_spec), np.amax(transformed_sound_spec))

    sound_spec = librosa.amplitude_to_db(sound_spec, ref=vmax)
    transformed_sound_spec = librosa.amplitude_to_db(transformed_sound_spec, ref=vmax)

    vmax = max(np.amax(sound_spec), np.amax(transformed_sound_spec))
    vmin = vmax - 85.0

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

    plt.tight_layout(pad=0.1)

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
            min_snr_db=5.0,
            max_snr_db=5.0,
            p=1.0,
        )

        sound, sample_rate = load_sound_file(
            librosa.example("libri1"), sample_rate=16000
        )

        sound = sound[..., 0 : int(4.7 * sample_rate)]

        transformed_sound = transform(sound, sample_rate)

        return sound, transformed_sound, sample_rate


@register
class AddGaussianNoiseExample(TransformUsageExample):
    transform_class = AddGaussianNoise

    def generate_example(self):
        random.seed(345)
        np.random.seed(345)
        transform = AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.01, p=1.0)

        sound, sample_rate = load_sound_file(
            librosa.example("libri1"), sample_rate=16000
        )
        sound = sound[..., 0 : int(4.7 * sample_rate)]

        transformed_sound = transform(sound, sample_rate)

        return sound, transformed_sound, sample_rate


@register
class AddGaussianSNRExample(TransformUsageExample):
    transform_class = AddGaussianSNR

    def generate_example(self):
        random.seed(345)
        np.random.seed(345)
        transform = AddGaussianSNR(min_snr_db=16.0, max_snr_db=16.0, p=1.0)

        sound, sample_rate = load_sound_file(
            librosa.example("libri1"), sample_rate=16000
        )
        sound = sound[..., 0 : int(4.7 * sample_rate)]

        transformed_sound = transform(sound, sample_rate)

        return sound, transformed_sound, sample_rate


@register
class AddShortNoisesExample(TransformUsageExample):
    transform_class = AddShortNoises

    def generate_example(self):
        random.seed(345)
        np.random.seed(345)
        transform = AddShortNoises(
            os.path.join(DEMO_DIR, "short_noises"),
            min_time_between_sounds=0.8,
            max_time_between_sounds=2.5,
            min_snr_db=0.0,
            max_snr_db=15.0,
            p=1.0,
        )

        sound, sample_rate = load_sound_file(
            librosa.example("pistachio"), sample_rate=None
        )
        sound = sound[..., 0 : int(4.6 * sample_rate)]

        transformed_sound = transform(sound, sample_rate)

        return sound, transformed_sound, sample_rate


@register
class AdjustDurationExample(TransformUsageExample):
    transform_class = AdjustDuration

    def generate_example(self):
        random.seed(42)
        np.random.seed(42)
        transform = AdjustDuration(duration_samples=60000, p=1.0)

        sound, sample_rate = load_sound_file(
            librosa.example("pistachio"), sample_rate=None
        )
        sound = sound[..., 0 : int(4.6 * sample_rate)]

        transformed_sound = transform(sound, sample_rate)

        return sound, transformed_sound, sample_rate


@register
class AirAbsorptionExample(TransformUsageExample):
    transform_class = AirAbsorption

    def generate_example(self):
        random.seed(42)
        np.random.seed(42)
        transform = AirAbsorption(
            min_temperature=20.0,
            max_temperature=20.0,
            min_humidity=70.0,
            max_humidity=70.0,
            min_distance=20.0,
            max_distance=20.0,
            p=1.0,
        )

        sound, sample_rate = load_sound_file(
            os.path.join(DEMO_DIR, "p286_011.wav"), sample_rate=None
        )
        sound = sound[..., int(0.5 * sample_rate) : int(2.9 * sample_rate)]

        transformed_sound = transform(sound, sample_rate)

        return sound, transformed_sound, sample_rate


@register
class ApplyImpulseResponseExample(TransformUsageExample):
    transform_class = ApplyImpulseResponse

    def generate_example(self):
        random.seed(42)
        np.random.seed(42)
        transform = ApplyImpulseResponse(
            ir_path=os.path.join(DEMO_DIR, "ir", "rir48000.wav"), p=1.0
        )

        sound, sample_rate = load_sound_file(
            os.path.join(DEMO_DIR, "p286_011.wav"), sample_rate=None
        )
        sound = sound[..., int(0.5 * sample_rate) : int(2.9 * sample_rate)]

        transformed_sound = transform(sound, sample_rate)

        return sound, transformed_sound, sample_rate


@register
class BandPassFilterExample(TransformUsageExample):
    transform_class = BandPassFilter

    def generate_example(self):
        random.seed(42)
        np.random.seed(42)
        transform = BandPassFilter(
            min_center_freq=2500.0,
            max_center_freq=2500.0,
            min_bandwidth_fraction=0.8,
            max_bandwidth_fraction=0.8,
            p=1.0,
        )

        sound, sample_rate = load_sound_file(
            os.path.join(DEMO_DIR, "p286_011.wav"), sample_rate=None
        )
        sound = sound[..., int(0.5 * sample_rate) : int(2.9 * sample_rate)]

        transformed_sound = transform(sound, sample_rate)

        return sound, transformed_sound, sample_rate


@register
class BandStopFilterExample(TransformUsageExample):
    transform_class = BandStopFilter

    def generate_example(self):
        random.seed(42)
        np.random.seed(42)
        transform = BandStopFilter(
            min_center_freq=2500.0,
            max_center_freq=2500.0,
            min_bandwidth_fraction=0.8,
            max_bandwidth_fraction=0.8,
            p=1.0,
        )

        sound, sample_rate = load_sound_file(
            os.path.join(DEMO_DIR, "p286_011.wav"), sample_rate=None
        )
        sound = sound[..., int(0.5 * sample_rate) : int(2.9 * sample_rate)]

        transformed_sound = transform(sound, sample_rate)

        return sound, transformed_sound, sample_rate


@register
class BitCrushExample(TransformUsageExample):
    transform_class = BitCrush

    def generate_example(self):
        random.seed(42)
        np.random.seed(42)
        transform = BitCrush(
            min_bit_depth=6,
            max_bit_depth=6,
            p=1.0,
        )

        sound, sample_rate = load_sound_file(
            os.path.join(DEMO_DIR, "p286_011.wav"), sample_rate=None
        )
        sound = sound[..., int(0.5 * sample_rate) : int(2.9 * sample_rate)]

        transformed_sound = transform(sound, sample_rate)

        return sound, transformed_sound, sample_rate


@register
class LimiterExample(TransformUsageExample):
    transform_class = Limiter

    def generate_example(self):
        random.seed(345)
        np.random.seed(345)
        transform = Limiter(min_threshold_db=-10, max_threshold_db=-10, p=1)

        sound, sample_rate = load_sound_file(
            librosa.example("libri1"), sample_rate=16000
        )
        sound = sound[..., 0 : int(4.7 * sample_rate)]

        transformed_sound = transform(sound, sample_rate)

        return sound, transformed_sound, sample_rate


@register
class PitchShiftExample(TransformUsageExample):
    transform_class = PitchShift

    def generate_example(self):
        random.seed(42)
        np.random.seed(42)
        transform = PitchShift(min_semitones=-4, max_semitones=-4, p=1.0)

        sound, sample_rate = load_sound_file(
            librosa.example("pistachio"), sample_rate=None
        )
        sound = sound[..., 0 : int(4.6 * sample_rate)]

        transformed_sound = transform(sound, sample_rate)

        return sound, transformed_sound, sample_rate


@register
class RepeatPartExample(TransformUsageExample):
    transform_class = RepeatPart

    def generate_example(self):
        random.seed(66)
        np.random.seed(66)
        transform = RepeatPart(
            min_part_duration=0.1,
            max_part_duration=0.5,
            crossfade_duration=0.03,
            part_transform=SevenBandParametricEQ(
                min_gain_db=-6.0, max_gain_db=6.0, p=1.0
            ),
            p=1.0,
        )

        sound, sample_rate = load_sound_file(
            os.path.join(DEMO_DIR, "p286_011.wav"), sample_rate=None
        )
        sound = sound[..., int(0.5 * sample_rate) : int(2.9 * sample_rate)]

        transformed_sound = transform(sound, sample_rate)

        return sound, transformed_sound, sample_rate


@register
class ReverseExample(TransformUsageExample):
    transform_class = Reverse

    def generate_example(self):
        transform = Reverse(p=1.0)

        sound, sample_rate = load_sound_file(
            os.path.join(DEMO_DIR, "p286_011.wav"), sample_rate=None
        )
        sound = sound[..., int(0.5 * sample_rate) : int(2.9 * sample_rate)]

        transformed_sound = transform(sound, sample_rate)

        return sound, transformed_sound, sample_rate


@register
class RoomSimulatorExample(TransformUsageExample):
    transform_class = RoomSimulator

    def generate_example(self):
        random.seed(345)
        np.random.seed(345)
        transform = RoomSimulator(p=1)

        sound, sample_rate = load_sound_file(
            librosa.example("libri1"), sample_rate=16000
        )
        sound = sound[..., 0 : int(4.7 * sample_rate)]

        transformed_sound = transform(sound, sample_rate)

        return sound, transformed_sound, sample_rate


@register
class SevenBandParametricEQExample(TransformUsageExample):
    transform_class = SevenBandParametricEQ

    def generate_example(self):
        random.seed(345)
        np.random.seed(345)
        transform = SevenBandParametricEQ(min_gain_db=3.0, max_gain_db=3.0, p=1)

        sound, sample_rate = load_sound_file(
            librosa.example("libri1"), sample_rate=16000
        )
        sound = sound[..., 0 : int(4.7 * sample_rate)]

        transformed_sound = transform(sound, sample_rate)

        return sound, transformed_sound, sample_rate


@register
class ShiftExample(TransformUsageExample):
    transform_class = Shift

    def generate_example(self):
        random.seed(345)
        np.random.seed(345)
        transform = Shift(min_shift=0.75, max_shift=0.75, rollover=True, p=1)

        sound, sample_rate = load_sound_file(
            librosa.example("libri1"), sample_rate=16000
        )
        sound = sound[..., 0 : int(4.7 * sample_rate)]

        transformed_sound = transform(sound, sample_rate)

        return sound, transformed_sound, sample_rate


@register
class TanhDistortionExample(TransformUsageExample):
    transform_class = TanhDistortion

    def generate_example(self):
        random.seed(345)
        np.random.seed(345)
        transform = TanhDistortion(min_distortion=0.25, max_distortion=0.25, p=1.0)

        sound, sample_rate = load_sound_file(
            librosa.example("libri1"), sample_rate=16000
        )
        sound = sound[..., 0 : int(4.7 * sample_rate)]

        transformed_sound = transform(sound, sample_rate)

        return sound, transformed_sound, sample_rate


@register
class TimeMaskExample(TransformUsageExample):
    transform_class = TimeMask

    def generate_example(self):
        random.seed(345)
        np.random.seed(345)
        transform = TimeMask(
            min_band_part=0.1,
            max_band_part=0.15,
            fade=True,
            p=1.0,
        )

        sound, sample_rate = load_sound_file(
            librosa.example("libri1"), sample_rate=16000
        )

        sound = sound[..., 0 : int(4.7 * sample_rate)]

        transformed_sound = transform(sound, sample_rate)

        return sound, transformed_sound, sample_rate


@register
class TimeStretchExample(TransformUsageExample):
    transform_class = TimeStretch

    def generate_example(self):
        random.seed(345)
        np.random.seed(345)
        transform = TimeStretch(
            min_rate=1.25,
            max_rate=1.25,
            leave_length_unchanged=True,
            p=1.0,
        )

        sound, sample_rate = load_sound_file(
            librosa.example("libri1"), sample_rate=16000
        )

        sound = sound[..., 0 : int(4.7 * sample_rate)]

        transformed_sound = transform(sound, sample_rate)

        return sound, transformed_sound, sample_rate


@register
class TrimExample(TransformUsageExample):
    transform_class = Trim

    def generate_example(self):
        random.seed(345)
        np.random.seed(345)
        transform = Trim(p=1.0)

        sound, sample_rate = load_sound_file(
            librosa.example("libri1"), sample_rate=16000
        )

        sound = sound[..., 0 : int(4.7 * sample_rate)]

        transformed_sound = transform(sound, sample_rate)

        return sound, transformed_sound, sample_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transform",
        dest="transform_name",
        type=str,
        required=True,
        choices=[c.__name__ for c in transform_usage_example_classes],
    )
    args = parser.parse_args()
    BASE_DIR = Path(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    for transform_class in transform_usage_example_classes:
        if transform_class.__name__.lower() != args.transform_name.lower():
            continue
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
