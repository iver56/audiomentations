import warnings

import librosa
import numpy as np


def load_sound_file(file_path, sample_rate, mono=True, resample_type="auto"):
    """
    Load an audio file as a floating point time series. Audio will be automatically
    resampled to the given sample rate.

    :param file_path: str or Path instance that points to a sound file
    :param sample_rate: If not None, resample to this sample rate
    :param mono: If True, mix any multichannel data down to mono, and return a 1D array
    :param resample_type: "auto" means use "kaiser_fast" when upsampling and "kaiser_best" when
        downsampling
    """
    file_path = str(file_path)
    samples, actual_sample_rate = librosa.load(
        str(file_path), sr=None, mono=mono, dtype=np.float32
    )

    if sample_rate is not None and actual_sample_rate != sample_rate:
        if resample_type == "auto":
            if librosa.__version__.startswith("0.8."):
                resample_type = (
                    "kaiser_fast" if actual_sample_rate < sample_rate else "kaiser_best"
                )
            else:
                resample_type = "soxr_hq"
        samples = librosa.resample(
            samples,
            orig_sr=actual_sample_rate,
            target_sr=sample_rate,
            res_type=resample_type,
        )
        warnings.warn(
            "{} had to be resampled from {} Hz to {} Hz. This hurt execution time.".format(
                str(file_path), actual_sample_rate, sample_rate
            )
        )

    actual_sample_rate = actual_sample_rate if sample_rate is None else sample_rate

    if mono:
        assert len(samples.shape) == 1
    return samples, actual_sample_rate
