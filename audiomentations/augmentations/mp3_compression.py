import os
import random
import tempfile
import uuid
import warnings
from typing import Literal

import librosa
import numpy as np
import sys
from numpy.typing import NDArray

from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import (
    convert_float_samples_to_int16,
    get_max_abs_amplitude,
)


class Mp3Compression(BaseWaveformTransform):
    """Compress the audio using an MP3 encoder to lower the audio quality.
    This may help machine learning models deal with compressed, low-quality audio.

    This transform depends on either lameenc or pydub/ffmpeg.

    Note that bitrates below 32 kbps are only supported for low sample rates (up to 24000 Hz).

    Note: When using the lameenc backend, the output may be slightly longer than the input due
    to the fact that the LAME encoder inserts some silence at the beginning of the audio.

    Warning: This transform writes to disk, so it may be slow. Ideally, the work should be done
    in memory. Contributions are welcome.
    """

    supports_multichannel = True

    SUPPORTED_BITRATES = [
        8,
        16,
        24,
        32,
        40,
        48,
        56,
        64,
        80,
        96,
        112,
        128,
        144,
        160,
        192,
        224,
        256,
        320,
    ]

    def __init__(
        self,
        min_bitrate: int = 8,
        max_bitrate: int = 64,
        backend: Literal["pydub", "lameenc", "fast-mp3-augment"] = "fast-mp3-augment",
        preserve_delay: bool = False,
        quality: int = 7,
        p: float = 0.5,
    ):
        """
        :param min_bitrate: Minimum bitrate in kbps
        :param max_bitrate: Maximum bitrate in kbps
        :param backend: "fast-mp3-augment", "pydub" or "lameenc".
            "fast-mp3-augment":
                * Fast (in-memory compute, encoder and decoder in separate threads working concurrently)
                * LAME encoder and minimp3 decoder
            "pydub":
                * Uses ffmpeg under the hood
                * Slower than lameenc and fast-mp3-augment
                * Writes temporary files to disk, which makes it comparatively slow
            "lameenc":
                * Slightly delays and pads the audio due to the way MP3 encoding and decoding normally works
                * Writes a temporary file to disk, which makes is comparatively slow
        :param preserve_delay:
            If False (default), the output length and timing will match the input.
            If True, include LAME encoder delay + filter delay (a few tens of milliseconds) and padding in the output.
            This makes the output
            1) longer than the input
            2) delayed (out of sync) relative to the input
            Normally, it makes sense to set preserve_delay to False, but if you want outputs that include the
            short, almost silent part in the beginning, you here have the option to get that.
        :param quality: LAME-specific parameter that controls a trade-off between audio quality and speed.
            quality is an int in range [0, 9]:
            0: higher quality audio at the cost of slower processing
            9: faster processing at the cost of lower quality audio
            Note: If using backend=="pydub", this parameter gets silently ignored.
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        if min_bitrate < self.SUPPORTED_BITRATES[0]:
            raise ValueError(
                "min_bitrate must be greater than or equal to"
                f" {self.SUPPORTED_BITRATES[0]}"
            )
        if max_bitrate > self.SUPPORTED_BITRATES[-1]:
            raise ValueError(
                "max_bitrate must be less than or equal to"
                f" {self.SUPPORTED_BITRATES[-1]}"
            )
        if max_bitrate < min_bitrate:
            raise ValueError("max_bitrate must be >= min_bitrate")

        is_any_supported_bitrate_in_range = any(
            min_bitrate <= bitrate <= max_bitrate for bitrate in self.SUPPORTED_BITRATES
        )
        if not is_any_supported_bitrate_in_range:
            raise ValueError(
                "There is no supported bitrate in the range between the specified"
                " min_bitrate and max_bitrate. The supported bitrates are:"
                f" {self.SUPPORTED_BITRATES}"
            )

        if backend == "pydub":
            if preserve_delay:
                raise ValueError(
                    'The "pydub" backend does not support preserve_delay=True. Switch to the'
                    ' "fast-mp3-augment" backend (recommended) or pass preserve_delay=False.'
                )
        elif backend == "lameenc":
            warnings.warn(
                'The "lameenc" backend is deprecated. Use backend="fast-mp3-augment" instead. It also uses'
                " the LAME encoder under the hood, but is faster.",
                DeprecationWarning,
            )
            if not preserve_delay:
                raise ValueError(
                    'The "lameenc" backend does not support preserve_delay=False. Switch to the'
                    ' "fast-mp3-augment" backend (recommended) or pass preserve_delay=True.'
                )
        self.preserve_delay = preserve_delay
        self.quality = quality
        self.min_bitrate = min_bitrate
        self.max_bitrate = max_bitrate
        if backend not in ("fast-mp3-augment", "pydub", "lameenc"):
            raise ValueError(
                'backend must be set to either "fast-mp3-augment", "pydub" or "lameenc"'
            )
        self.backend = backend
        self.post_gain_factor = None

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            bitrate_choices = [
                bitrate
                for bitrate in self.SUPPORTED_BITRATES
                if self.min_bitrate <= bitrate <= self.max_bitrate
            ]
            self.parameters["bitrate"] = random.choice(bitrate_choices)

    def apply(
        self, samples: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        if self.backend == "fast-mp3-augment":
            return self.apply_fast_mp3_augment(samples, sample_rate)
        if self.backend == "lameenc":
            return self.apply_lameenc(samples, sample_rate)
        elif self.backend == "pydub":
            return self.apply_pydub(samples, sample_rate)
        else:
            raise Exception("Backend {} not recognized".format(self.backend))

    def maybe_pre_gain(self, samples):
        """
        If the audio is too loud, gain it down to avoid distortion in the audio file to
        be encoded.
        """
        greatest_abs_sample = get_max_abs_amplitude(samples)
        if greatest_abs_sample > 1.0:
            self.post_gain_factor = greatest_abs_sample
            samples = samples * (1.0 / greatest_abs_sample)
        else:
            self.post_gain_factor = None
        return samples

    def maybe_post_gain(self, samples):
        """If the audio was pre-gained down earlier, post-gain it up to compensate here."""
        if self.post_gain_factor is not None:
            samples = samples * self.post_gain_factor
        return samples

    def apply_lameenc(
        self, samples: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        try:
            import lameenc
        except ImportError:
            print(
                (
                    "Failed to import the lame encoder. Maybe it is not installed? "
                    "To install the optional lameenc dependency of audiomentations,"
                    " do `pip install audiomentations[extras]` or simply"
                    " `pip install lameenc`"
                ),
                file=sys.stderr,
            )
            raise

        assert samples.dtype == np.float32

        samples = self.maybe_pre_gain(samples)

        int_samples = convert_float_samples_to_int16(samples).T

        num_channels = 1 if samples.ndim == 1 else samples.shape[0]

        encoder = lameenc.Encoder()
        encoder.set_bit_rate(self.parameters["bitrate"])
        encoder.set_in_sample_rate(sample_rate)
        encoder.set_channels(num_channels)
        encoder.set_quality(self.quality)
        encoder.silence()

        mp3_data = encoder.encode(int_samples.tobytes())
        mp3_data += encoder.flush()

        # Write a temporary MP3 file that will then be decoded
        tmp_dir = tempfile.gettempdir()
        tmp_file_path = os.path.join(
            tmp_dir, "tmp_compressed_{}.mp3".format(str(uuid.uuid4())[0:12])
        )
        with open(tmp_file_path, "wb") as f:
            f.write(mp3_data)

        degraded_samples, _ = librosa.load(tmp_file_path, sr=sample_rate, mono=False)

        os.unlink(tmp_file_path)

        degraded_samples = self.maybe_post_gain(degraded_samples)

        if num_channels == 1:
            if int_samples.ndim == 1 and degraded_samples.ndim == 2:
                degraded_samples = np.ravel(degraded_samples)
            elif int_samples.ndim == 2 and degraded_samples.ndim == 1:
                degraded_samples = degraded_samples.reshape((1, -1))

        return degraded_samples

    def apply_fast_mp3_augment(
        self, samples: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        try:
            import fast_mp3_augment
        except ImportError:
            print(
                (
                    "Failed to import fast_mp3_augment. Maybe it is not installed? "
                    "To install the optional fast_mp3_augment dependency of audiomentations,"
                    " do `pip install audiomentations[extras]` or simply"
                    " `pip install fast_mp3_augment`"
                ),
                file=sys.stderr,
            )
            raise

        assert samples.dtype == np.float32

        if samples.ndim == 2 and not samples.flags.c_contiguous:
            samples = np.ascontiguousarray(samples)

        degraded_samples = fast_mp3_augment.compress_roundtrip(
            samples,
            sample_rate=sample_rate,
            bitrate_kbps=self.parameters["bitrate"],
            preserve_delay=self.preserve_delay,
            quality=self.quality,
        )
        return degraded_samples

    def apply_pydub(
        self, samples: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
        try:
            import pydub
        except ImportError:
            print(
                (
                    "Failed to import pydub. Maybe it is not installed? "
                    "To install the optional pydub dependency of audiomentations,"
                    " do `pip install audiomentations[extras]` or simply"
                    " `pip install pydub`"
                ),
                file=sys.stderr,
            )
            raise

        assert samples.dtype == np.float32

        samples = self.maybe_pre_gain(samples)

        int_samples = convert_float_samples_to_int16(samples).T
        num_channels = 1 if samples.ndim == 1 else samples.shape[0]
        audio_segment = pydub.AudioSegment(
            int_samples.tobytes(),
            frame_rate=sample_rate,
            sample_width=int_samples.dtype.itemsize,
            channels=num_channels,
        )

        tmp_dir = tempfile.gettempdir()
        tmp_file_path = os.path.join(
            tmp_dir, "tmp_compressed_{}.mp3".format(str(uuid.uuid4())[0:12])
        )

        bitrate_string = "{}k".format(self.parameters["bitrate"])
        file_handle = audio_segment.export(tmp_file_path, bitrate=bitrate_string)
        file_handle.close()

        degraded_samples, _ = librosa.load(tmp_file_path, sr=sample_rate, mono=False)

        os.unlink(tmp_file_path)

        degraded_samples = self.maybe_post_gain(degraded_samples)

        if num_channels == 1:
            if int_samples.ndim == 1 and degraded_samples.ndim == 2:
                degraded_samples = np.ravel(degraded_samples)
            elif int_samples.ndim == 2 and degraded_samples.ndim == 1:
                degraded_samples = degraded_samples.reshape((1, -1))

        return degraded_samples
