import os
import random
import time

import numpy as np
from scipy.io import wavfile

from audiomentations import (
    Compose,
    AddGaussianNoise,
    TimeStretch,
    PitchShift,
    Shift,
    Normalize,
    AddImpulseResponse,
    FrequencyMask,
    TimeMask,
    AddGaussianSNR,
    Resample,
    ClippingDistortion,
    AddBackgroundNoise,
    AddShortNoises,
    PolarityInversion,
    Gain,
)

SAMPLE_RATE = 16000
CHANNELS = 1
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


def load_wav_file(sound_file_path):
    """Load the wav file at the given file path and return a float32 numpy array."""
    sample_rate, sound_np = wavfile.read(sound_file_path)
    if sample_rate != SAMPLE_RATE:
        raise Exception(
            "Unexpected sample rate {} (expected {})".format(sample_rate, SAMPLE_RATE)
        )

    if sound_np.dtype != np.float32:
        assert sound_np.dtype == np.int16
        sound_np = np.divide(
            sound_np, 32768, dtype=np.float32
        )  # ends up roughly between -1 and 1

    return sound_np


if __name__ == "__main__":
    """
    For each transformation, apply it to an example sound and write the transformed sounds to
    an output folder. Also crudely measure and print execution time.
    """
    output_dir = os.path.join(DEMO_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)

    np.random.seed(42)
    random.seed(42)

    samples = load_wav_file(os.path.join(DEMO_DIR, "acoustic_guitar_0.wav"))

    transforms = [
        {
            "instance": AddImpulseResponse(p=1.0, ir_path=os.path.join(DEMO_DIR, "ir")),
            "num_runs": 1,
        },
        {"instance": FrequencyMask(p=1.0), "num_runs": 5},
        {"instance": TimeMask(p=1.0), "num_runs": 5},
        {"instance": AddGaussianSNR(p=1.0), "num_runs": 5},
        {
            "instance": AddGaussianNoise(
                min_amplitude=0.001, max_amplitude=0.015, p=1.0
            ),
            "num_runs": 5,
        },
        {"instance": TimeStretch(min_rate=0.8, max_rate=1.25, p=1.0), "num_runs": 5},
        {
            "instance": PitchShift(min_semitones=-4, max_semitones=4, p=1.0),
            "num_runs": 5,
        },
        {"instance": Shift(min_fraction=-0.5, max_fraction=0.5, p=1.0), "num_runs": 5},
        {
            "instance": Shift(
                min_fraction=-0.5, max_fraction=0.5, rollover=False, p=1.0
            ),
            "num_runs": 5,
            "name": "ShiftWithoutRollover",
        },
        {"instance": Normalize(p=1.0), "num_runs": 1},
        {"instance": Resample(p=1.0), "num_runs": 5},
        {"instance": ClippingDistortion(p=1.0), "num_runs": 5},
        {
            "instance": AddBackgroundNoise(
                sounds_path=os.path.join(DEMO_DIR, "background_noises"), p=1.0
            ),
            "num_runs": 5,
        },
        {
            "instance": AddShortNoises(
                sounds_path=os.path.join(DEMO_DIR, "short_noises"),
                min_snr_in_db=0,
                max_snr_in_db=8,
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
        },
        {"instance": PolarityInversion(p=1.0), "num_runs": 1},
        {"instance": Gain(min_gain_in_db=-6, max_gain_in_db=6, p=1.0), "num_runs": 1},
    ]

    execution_times = {}

    for transform in transforms:
        augmenter = Compose([transform["instance"]])
        run_name = (
            transform.get("name")
            if transform.get("name")
            else transform["instance"].__class__.__name__
        )
        execution_times[run_name] = []
        for i in range(transform["num_runs"]):
            output_file_path = os.path.join(
                output_dir, "{}_{:03d}.wav".format(run_name, i)
            )
            with timer() as t:
                augmented_samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
            execution_times[run_name].append(t.execution_time)
            wavfile.write(output_file_path, rate=SAMPLE_RATE, data=augmented_samples)

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
            print("{:<32} {:.3f} s".format(run_name, np.mean(execution_times[run_name])))
