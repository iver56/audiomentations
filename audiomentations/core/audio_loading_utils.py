import warnings
import librosa
import numpy as np


# =============================================================================
# Audio Cache Backend
# =============================================================================
# A cache backend with get/put interface for use with Ray Actors or similar.
# The backend must implement:
#   - get(key) -> np.ndarray or None
#   - put(key, audio) -> None
# Key format: (file_path, sample_rate, mono)

_audio_cache_backend = None


def set_audio_cache_backend(backend):
    """
    Set a cache backend that load_sound_file() will use for caching audio.
    
    The backend must implement:
        - get(key) -> np.ndarray or None
        - put(key, audio) -> None
    
    Key format: (file_path: str, sample_rate: int, mono: bool)
    
    This enables mutable caching with LRU eviction, memory limits, etc.
    For Ray, use an Actor that implements this interface.
    
    Args:
        backend: Object with get/put methods, or None to disable.
    """
    global _audio_cache_backend
    _audio_cache_backend = backend


def get_audio_cache_backend():
    """Get the current audio cache backend, or None if not set."""
    return _audio_cache_backend


def clear_audio_cache_backend():
    """Clear the audio cache backend."""
    global _audio_cache_backend
    _audio_cache_backend = None


# =============================================================================
# Audio Loading
# =============================================================================

def _load_and_resample(file_path, sample_rate, mono, resample_type, offset=0.0, duration=None):
    """Internal: load from disk and resample. Returns (samples, actual_sample_rate)."""
    samples, actual_sample_rate = librosa.load(
        str(file_path), sr=None, mono=mono, dtype=np.float32, offset=offset, duration=duration
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
        actual_sample_rate = sample_rate

    if mono:
        assert len(samples.shape) == 1
    
    return samples, actual_sample_rate


def _slice_audio(samples, sample_rate, offset, duration):
    """Slice audio by offset/duration."""
    if offset > 0.0 or duration is not None:
        start_sample = int(offset * sample_rate)
        if duration is not None:
            end_sample = start_sample + int(duration * sample_rate)
            return samples[..., start_sample:end_sample]
        else:
            return samples[..., start_sample:]
    return samples


def load_sound_file(file_path, sample_rate, mono=True, resample_type="auto", offset=0.0, duration=None):
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
    cache_key = (file_path, sample_rate, mono)
    
    # Check cache backend (mutable cache with get/put)
    if _audio_cache_backend is not None:
        cached = _audio_cache_backend.get(cache_key)
        if cached is not None:
            # Cache hit: slice and return
            sliced = _slice_audio(cached, sample_rate, offset, duration)
            return sliced.copy(), sample_rate
        else:
            # Cache miss: load FULL file, cache it, then slice
            full_samples, actual_sr = _load_and_resample(file_path, sample_rate, mono, resample_type)
            _audio_cache_backend.put(cache_key, full_samples)
            sliced = _slice_audio(full_samples, actual_sr, offset, duration)
            return sliced.copy(), actual_sr
    
    # No cache: load with offset/duration directly (more efficient for one-off loads)
    samples, actual_sr = _load_and_resample(file_path, sample_rate, mono, resample_type, offset, duration)
    return samples, actual_sr
