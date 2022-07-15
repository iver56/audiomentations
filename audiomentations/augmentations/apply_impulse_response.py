import functools
import random
import warnings
from pathlib import Path
from typing import Optional, List, Union

import numpy as np
from scipy.signal import convolve

from audiomentations.core.audio_loading_utils import load_sound_file
from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import find_audio_files_in_paths


class ApplyImpulseResponse(BaseWaveformTransform):
    """Convolve the audio with a random impulse response.
    Impulse responses can be created using e.g. http://tulrich.com/recording/ir_capture/
    Impulse responses are represented as wav files in the given ir_path.
    """

    supports_multichannel = True

    def __init__(
        self,
        ir_path: Union[List[Path], List[str], str, Path],
        p=0.5,
        lru_cache_size=128,
        leave_length_unchanged: Optional[bool] = None,
    ):
        """
        :param ir_path: A path or list of paths to audio file(s) and/or folder(s) with
            audio files. Can be str or Path instance(s). The audio files given here are
            supposed to be impulse responses.
        :param p: The probability of applying this transform
        :param lru_cache_size: Maximum size of the LRU cache for storing impulse response files
        in memory.
        :param leave_length_unchanged: When set to True, the tail of the sound (e.g. reverb at
            the end) will be chopped off so that the length of the output is equal to the
            length of the input.
        """
        super().__init__(p)
        self.ir_files = find_audio_files_in_paths(ir_path)
        self.ir_files = [str(p) for p in self.ir_files]
        assert len(self.ir_files) > 0
        self.__load_ir = functools.lru_cache(maxsize=lru_cache_size)(
            ApplyImpulseResponse.__load_ir
        )
        if leave_length_unchanged is None:
            warnings.warn(
                "The default value of leave_length_unchanged will change from False to"
                " True in a future version of audiomentations. You can set the value"
                " explicitly to remove this warning for now.",
                FutureWarning
            )
            leave_length_unchanged = False

        self.leave_length_unchanged = leave_length_unchanged

    @staticmethod
    def __load_ir(file_path, sample_rate):
        return load_sound_file(file_path, sample_rate)

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["ir_file_path"] = random.choice(self.ir_files)

    def apply(self, samples, sample_rate):
        ir, sample_rate2 = self.__load_ir(self.parameters["ir_file_path"], sample_rate)
        if sample_rate != sample_rate2:
            # This will typically not happen, as librosa should automatically resample the
            # impulse response sound to the desired sample rate
            raise Exception(
                "Recording sample rate {} did not match Impulse Response signal"
                " sample rate {}!".format(sample_rate, sample_rate2)
            )

        if samples.ndim > 1:
            signal_ir = []
            for i in range(samples.shape[0]):
                channel_conv = convolve(samples[i], ir)
                signal_ir.append(channel_conv)
            signal_ir = np.array(signal_ir, dtype=samples.dtype)
        else:
            signal_ir = convolve(samples, ir)

        max_value = max(np.amax(signal_ir), -np.amin(signal_ir))
        if max_value > 0.0:
            scale = 0.5 / max_value
            signal_ir *= scale
        if self.leave_length_unchanged:
            signal_ir = signal_ir[..., : samples.shape[-1]]
        return signal_ir

    def __getstate__(self):
        state = self.__dict__.copy()
        warnings.warn(
            "Warning: the LRU cache of ApplyImpulseResponse gets discarded when pickling it."
            " E.g. this means the cache will be not be used when using ApplyImpulseResponse"
            " together with multiprocessing on Windows"
        )
        del state["_ApplyImpulseResponse__load_ir"]
        return state
