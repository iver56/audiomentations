import functools
import random
import warnings
from pathlib import Path
from typing import List, Union

import numpy as np
from numpy.typing import NDArray
from scipy.signal import convolve
import itertools

from audiomentations.core.audio_loading_utils import load_sound_file
from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import find_audio_files_in_paths


class ApplyImpulseResponse(BaseWaveformTransform):
    """Convolve the audio with a randomly selected impulse response.
    Impulse responses can be created using e.g. http://tulrich.com/recording/ir_capture/
    Impulse responses are represented as audio (ideally wav) files in the given ir_path.
    """

    supports_multichannel = True

    def __init__(
        self,
        ir_path: Union[List[Path], List[str], str, Path],
        p=0.5,
        lru_cache_size=None,
        leave_length_unchanged: bool = True,
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
        self.ir_path = ir_path
        self.ir_files = [str(p) for p in find_audio_files_in_paths(self.ir_path)]
        assert self.ir_files, "No impulse response files found at the specified path."
        if lru_cache_size is not None:
            raise ValueError(
                "Passing lru_cache_size is no longer supported, as the cache has been removed (since v0.43.0)."
            )
        self.leave_length_unchanged = leave_length_unchanged

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["ir_file_path"] = random.choice(self.ir_files)            

    def apply(self, samples: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
        # Determine if the impulse response should be loaded as mono
        load_mono_ir = samples.ndim == 1
        # Load the impulse response file and cut the tail if necessary
        rir_duration = None if self.leave_length_unchanged else len(samples) / sample_rate
        ir, sample_rate2 = load_sound_file(self.parameters["ir_file_path"], 
                                           sample_rate, 
                                           mono=load_mono_ir, 
                                           offset=0.0, # always start at the beginning
                                           duration=rir_duration)
        if sample_rate != sample_rate2:
            # This will typically not happen, as librosa should automatically resample the
            # impulse response sound to the desired sample rate
            raise Exception(
                "Recording sample rate {} did not match Impulse Response signal"
                " sample rate {}!".format(sample_rate, sample_rate2)
            )

        # Expand dimensions to match
        samples_original_dim = samples.ndim
        samples, ir = np.atleast_2d(samples), np.atleast_2d(ir)

        # Preallocate the output array
        output_shape = (samples.shape[0], samples.shape[1] + ir.shape[1] - 1)
        signal_ir = np.empty(output_shape, dtype=samples.dtype)

        # Loop over all samples channels for channelwise convolution
        for i, (sample, impulse_response) in enumerate(zip(samples, itertools.cycle(ir))):
            signal_ir[i, :] = convolve(sample, impulse_response)

        max_value = max(np.amax(signal_ir), -np.amin(signal_ir))
        if max_value > 0.0:
            scale = 0.5 / max_value
            signal_ir *= scale
        if self.leave_length_unchanged:
            signal_ir = signal_ir[..., : samples.shape[-1]]

        # reshape if mono input
        if samples_original_dim == 1:
            signal_ir = signal_ir[0]

        return signal_ir
