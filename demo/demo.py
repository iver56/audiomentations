import os
import random
from pathlib import Path

import numpy as np
import time
from scipy.io import wavfile

from audiomentations import (
    AddGaussianNoise,
    TimeStretch,
    PitchShift,
    Shift,
    Normalize,
    FrequencyMask,
    TimeMask,
    AddGaussianSNR,
    Resample,
    ClippingDistortion,
    AddBackgroundNoise,
    AddShortNoises,
    PeakingFilter,
    PolarityInversion,
    Gain,
    Mp3Compression,
    LoudnessNormalization,
    Trim,
    LowPassFilter,
    LowShelfFilter,
    HighShelfFilter,
    HighPassFilter,
    BandPassFilter,
    ApplyImpulseResponse,
    Reverse,
    TanhDistortion,
    Compose,
    SomeOf,
    OneOf,
    BandStopFilter,
    GainTransition,
)
from audiomentations.core.audio_loading_utils import load_sound_file
from audiomentations.core.transforms_interface import (
    MultichannelAudioNotSupportedException,
)

DEMO_DIR = os.path.dirname(__file__)


class timer(object):
    """
    timer: A class used to measure the execution time of a block of code that is
    inside a "with" statement.

    Example:

    ```
    with timer("Count to 500000"):
        x = 0
        for i in range(500000):
            x += 1
        print(x)
    ```

    Will output:
    500000
    Count to 500000: 0.04 s

    Warning: The time resolution used here may be limited to 1 ms
    """

    def __init__(self, description="Execution time", verbose=False):
        self.description = description
        self.verbose = verbose
        self.execution_time = None

    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.execution_time = time.time() - self.t
        if self.verbose:
            print("{}: {:.3f} s".format(self.description, self.execution_time))


if __name__ == "__main__":
    """
    For each transformation, apply it to an example sound and write the transformed sounds to
    an output folder. Also crudely measure and print execution time.
    """
    output_dir = os.path.join(DEMO_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)

    np.random.seed(420)
    random.seed(420)

    sound_file_paths = [
        Path(os.path.join(DEMO_DIR, "acoustic_guitar_0.wav")),
        Path(os.path.join(DEMO_DIR, "perfect-alley1.ogg")),
        Path(os.path.join(DEMO_DIR, "p286_011.wav")),
    ]

    transforms = [
        {
            "instance": AddBackgroundNoise(
                sounds_path=os.path.join(DEMO_DIR, "background_noises"), p=1.0
            ),
            "num_runs": 5,
            "name": "AddBackgroundNoiseRelative",
        },
        {
            "instance": AddBackgroundNoise(
                sounds_path=os.path.join(DEMO_DIR, "background_noises"),
                noise_rms="absolute",
                min_absolute_rms_in_db=-30,
                max_absolute_rms_in_db=-10,
                p=1.0,
            ),
            "num_runs": 5,
            "name": "AddBackgroundNoiseAbsolute",
        },
        {
            "instance": AddGaussianNoise(
                min_amplitude=0.001, max_amplitude=0.015, p=1.0
            ),
            "num_runs": 5,
        },
        {
            "instance": AddGaussianSNR(p=1.0),
            "num_runs": 5,
            "name": "AddGaussianSNRLegacy",
        },
        {
            "instance": AddGaussianSNR(min_snr_in_db=0, max_snr_in_db=35, p=1.0),
            "num_runs": 5,
            "name": "AddGaussianSNRNew",
        },
        {
            "instance": ApplyImpulseResponse(
                p=1.0, ir_path=os.path.join(DEMO_DIR, "ir")
            ),
            "num_runs": 1,
        },
        {
            "instance": ApplyImpulseResponse(
                p=1.0, ir_path=os.path.join(DEMO_DIR, "ir"), leave_length_unchanged=True
            ),
            "num_runs": 1,
            "name": "AddImpulseResponseLeaveLengthUnchanged",
        },
        {
            "instance": AddShortNoises(
                sounds_path=os.path.join(DEMO_DIR, "short_noises"),
                min_snr_in_db=0,
                max_snr_in_db=8,
                min_absolute_noise_rms_db=-30,
                max_absolute_noise_rms_db=-10,
                noise_rms="absolute",
                min_time_between_sounds=2.0,
                max_time_between_sounds=4.0,
                burst_probability=0.4,
                min_pause_factor_during_burst=0.01,
                max_pause_factor_during_burst=0.95,
                min_fade_in_time=0.005,
                max_fade_in_time=0.08,
                min_fade_out_time=0.01,
                max_fade_out_time=0.1,
                p=1.0,
            ),
            "num_runs": 5,
            "name": "AddShortNoisesAbsolute",
        },
        {
            "instance": AddShortNoises(
                sounds_path=os.path.join(DEMO_DIR, "short_noises"),
                min_snr_in_db=0,
                max_snr_in_db=8,
                min_absolute_noise_rms_db=-30,
                max_absolute_noise_rms_db=-10,
                noise_rms="relative",
                min_time_between_sounds=2.0,
                max_time_between_sounds=4.0,
                burst_probability=0.4,
                min_pause_factor_during_burst=0.01,
                max_pause_factor_during_burst=0.95,
                min_fade_in_time=0.005,
                max_fade_in_time=0.08,
                min_fade_out_time=0.01,
                max_fade_out_time=0.1,
                p=1.0,
            ),
            "num_runs": 5,
            "name": "AddShortNoisesRelative",
        },
        {"instance": BandPassFilter(p=1.0), "num_runs": 5},
        {"instance": BandStopFilter(p=1.0), "num_runs": 5},
        {"instance": ClippingDistortion(p=1.0), "num_runs": 5},
        {
            "instance": FrequencyMask(
                min_frequency_band=0.5, max_frequency_band=0.6, p=1.0
            ),
            "num_runs": 5,
        },
        {"instance": Gain(min_gain_in_db=-6, max_gain_in_db=6, p=1.0), "num_runs": 5},
        {"instance": GainTransition(p=1.0), "num_runs": 5},
        {"instance": HighPassFilter(p=1.0), "num_runs": 5},
        {"instance": HighShelfFilter(p=1.0), "num_runs": 5},
        {"instance": LowPassFilter(p=1.0), "num_runs": 5},
        {"instance": LowShelfFilter(p=1.0), "num_runs": 5},
        {
            "instance": PitchShift(min_semitones=-4, max_semitones=4, p=1.0),
            "num_runs": 5,
        },
        {"instance": LoudnessNormalization(p=1.0), "num_runs": 5},
        {
            "instance": Mp3Compression(backend="lameenc", p=1.0),
            "num_runs": 5,
            "name": "Mp3CompressionLameenc",
        },
        {
            "instance": Mp3Compression(backend="pydub", p=1.0),
            "num_runs": 5,
            "name": "Mp3CompressionPydub",
        },
        {"instance": Normalize(p=1.0), "num_runs": 1},
        {"instance": PeakingFilter(p=1.0), "num_runs": 5},
        {"instance": PolarityInversion(p=1.0), "num_runs": 1},
        {"instance": Resample(p=1.0), "num_runs": 5},
        {"instance": Reverse(p=1.0), "num_runs": 1},
        {
            "instance": Shift(min_fraction=-0.5, max_fraction=0.5, fade=False, p=1.0),
            "num_runs": 5,
            "name": "ShiftWithoutFade",
        },
        {
            "instance": Shift(min_fraction=-0.5, max_fraction=0.5, fade=True, p=1.0),
            "num_runs": 5,
            "name": "ShiftWithShortFade",
        },
        {
            "instance": Shift(
                min_fraction=-0.5,
                max_fraction=0.5,
                rollover=False,
                fade=True,
                fade_duration=0.3,
                p=1.0,
            ),
            "num_runs": 5,
            "name": "ShiftWithoutRolloverWithLongFade",
        },
        {"instance": TanhDistortion(p=1.0), "num_runs": 5},
        {"instance": TimeMask(p=1.0), "num_runs": 5},
        {"instance": TimeStretch(min_rate=0.8, max_rate=1.25, p=1.0), "num_runs": 5},
        {"instance": Trim(p=1.0), "num_runs": 1},
        {
            "instance": Compose(
                [
                    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                    SomeOf(
                        (0, 2),
                        [
                            TimeStretch(min_rate=0.8, max_rate=1.25, p=1.0),
                            PitchShift(min_semitones=-4, max_semitones=4, p=1.0),
                        ],
                    ),
                    Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
                    OneOf([TanhDistortion(p=1.0), ClippingDistortion(p=1.0)], p=0.25),
                ]
            ),
            "num_runs": 10,
            "name": "BigCompose",
        },
    ]

    for sound_file_path in sound_file_paths:
        samples, sample_rate = load_sound_file(
            sound_file_path, sample_rate=None, mono=False
        )
        if len(samples.shape) == 2 and samples.shape[0] > samples.shape[1]:
            samples = samples.transpose()

        print(
            "Transforming {} with shape {}".format(
                sound_file_path.name, str(samples.shape)
            )
        )
        execution_times = {}

        for transform in transforms:
            augmenter = transform["instance"]
            run_name = (
                transform.get("name")
                if transform.get("name")
                else transform["instance"].__class__.__name__
            )
            execution_times[run_name] = []
            for i in range(transform["num_runs"]):
                output_file_path = os.path.join(
                    output_dir,
                    "{}_{}_{:03d}.wav".format(sound_file_path.stem, run_name, i),
                )
                try:
                    with timer() as t:
                        augmented_samples = augmenter(
                            samples=samples, sample_rate=sample_rate
                        )
                    execution_times[run_name].append(t.execution_time)

                    if len(augmented_samples.shape) == 2:
                        augmented_samples = augmented_samples.transpose()

                    wavfile.write(
                        output_file_path, rate=sample_rate, data=augmented_samples
                    )
                except MultichannelAudioNotSupportedException as e:
                    print(e)

        for run_name in execution_times:
            if len(execution_times[run_name]) > 1:
                print(
                    "{:<32} {:.3f} s (std: {:.3f} s)".format(
                        run_name,
                        np.mean(execution_times[run_name]),
                        np.std(execution_times[run_name]),
                    )
                )
            else:
                print(
                    "{:<32} {:.3f} s".format(
                        run_name, np.mean(execution_times[run_name])
                    )
                )
