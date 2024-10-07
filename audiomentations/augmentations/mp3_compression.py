import os
import random
import tempfile
import uuid

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
        backend: str = "pydub",
        p: float = 0.5,
    ):
        """
        :param min_bitrate: Minimum bitrate in kbps
        :param max_bitrate: Maximum bitrate in kbps
        :param backend: "pydub" or "lameenc".
            Pydub may use ffmpeg under the hood.
                Pros: Seems to avoid introducing latency in the output.
                Cons: Slower than lameenc.
            lameenc:
                Pros: You can set the quality parameter in addition to bitrate.
                Cons: Seems to introduce some silence at the start of the audio.
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

        self.min_bitrate = min_bitrate
        self.max_bitrate = max_bitrate
        if backend not in ("pydub", "lameenc"):
            raise ValueError('backend must be set to either "pydub" or "lameenc"')
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
        encoder.set_quality(7)  # 2 = highest, 7 = fastest
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
