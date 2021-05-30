import warnings
from pathlib import Path

import librosa
import numpy as np
from scipy.io import wavfile

IS_WAVIO_INSTALLED = True
try:
    import wavio
except ImportError:
    IS_WAVIO_INSTALLED = False


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
    file_path = Path(file_path)
    if file_path.name.lower().endswith(".wav"):
        # Use scipy for loading most wav files, because scipy is fast
        try:
            return load_wav_file(
                file_path, sample_rate, mono, resample_type=resample_type
            )
        except Exception as e:
            # scipy<1.6.0 does not natively support 24-bit wavs, so we use wavio or librosa.
            if "the wav file has 24-bit data" in str(e):
                if IS_WAVIO_INSTALLED:
                    return load_wav_file_with_wavio(
                        file_path, sample_rate, mono, resample_type=resample_type
                    )
                else:
                    warnings.warn(
                        "You are loading a 24-bit wav file, and librosa is not very fast at"
                        " doing that. Install wavio for a performance boost. To install the"
                        " optional wavio dependency of audiomentations,"
                        " do `pip install audiomentations[extras]` instead of"
                        " `pip install audiomentations`"
                    )
            elif "Unknown wave file format" in str(e):
                # This can happen if the file is in MS ADPCM format
                pass
            else:
                raise e
    samples, actual_sample_rate = librosa.load(
        str(file_path), sr=None, mono=mono, dtype=np.float32
    )

    if sample_rate is not None and actual_sample_rate != sample_rate:
        if resample_type == "auto":
            resample_type = (
                "kaiser_fast" if actual_sample_rate < sample_rate else "kaiser_best"
            )
        samples = librosa.resample(
            samples, actual_sample_rate, sample_rate, res_type=resample_type
        )
        warnings.warn(
            "{} had to be resampled from {} hz to {} hz. This hurt execution time.".format(
                str(file_path), actual_sample_rate, sample_rate
            )
        )

    actual_sample_rate = actual_sample_rate if sample_rate is None else sample_rate

    if mono:
        assert len(samples.shape) == 1
    return samples, actual_sample_rate


def load_wav_file(file_path, sample_rate, mono=True, resample_type="kaiser_best"):
    """Load a wav audio file as a floating point time series. Significantly faster than
    load_sound_file."""

    actual_sample_rate, samples = wavfile.read(file_path)

    if samples.dtype == np.float64:
        samples = samples.astype(np.float32)
        
    if samples.dtype != np.float32:
        if samples.dtype == np.int16:
            samples = np.true_divide(
                samples, 32768, dtype=np.float32
            )  # ends up roughly between -1 and 1
        elif samples.dtype == np.int32:
            samples = np.true_divide(
                samples, 2147483648, dtype=np.float32
            )  # ends up roughly between -1 and 1
        else:
            # TODO: Add support for 24-bit loading in scipy>=1.6.0
            raise Exception("Unexpected data type")

    if mono and len(samples.shape) > 1:
        if samples.shape[1] == 1:
            samples = samples[:, 0]
        else:
            samples = np.mean(samples, axis=1)

    if sample_rate is not None and actual_sample_rate != sample_rate:
        if resample_type == "auto":
            resample_type = (
                "kaiser_fast" if actual_sample_rate < sample_rate else "kaiser_best"
            )

        samples = librosa.resample(
            samples, actual_sample_rate, sample_rate, res_type=resample_type
        )
        warnings.warn(
            "{} had to be resampled from {} hz to {} hz. This hurt execution time.".format(
                str(file_path), actual_sample_rate, sample_rate
            )
        )

    actual_sample_rate = actual_sample_rate if sample_rate is None else sample_rate

    return samples, actual_sample_rate


def load_wav_file_with_wavio(
    file_path, sample_rate, mono=True, resample_type="kaiser_best"
):
    """Load a 24-bit wav audio file as a floating point time series. Significantly faster than
    load_sound_file."""

    wavio_obj = wavio.read(str(file_path))
    samples = wavio_obj.data
    actual_sample_rate = wavio_obj.rate

    if samples.dtype != np.float32:
        if wavio_obj.sampwidth == 3:
            samples = np.true_divide(
                samples, 8388608, dtype=np.float32
            )  # ends up roughly between -1 and 1
        elif wavio_obj.sampwidth == 2:
            samples = np.true_divide(
                samples, 32768, dtype=np.float32
            )  # ends up roughly between -1 and 1
        else:
            raise Exception("Unknown sampwidth")

    if mono and len(samples.shape) > 1:
        if samples.shape[1] == 1:
            samples = samples[:, 0]
        else:
            samples = np.mean(samples, axis=1)

    if sample_rate is not None and actual_sample_rate != sample_rate:
        if resample_type == "auto":
            resample_type = (
                "kaiser_fast" if actual_sample_rate < sample_rate else "kaiser_best"
            )
        samples = librosa.resample(
            samples, actual_sample_rate, sample_rate, res_type=resample_type
        )
        warnings.warn(
            "{} had to be resampled from {} hz to {} hz. This hurt execution time.".format(
                str(file_path), actual_sample_rate, sample_rate
            )
        )

    actual_sample_rate = actual_sample_rate if sample_rate is None else sample_rate

    return samples, actual_sample_rate
