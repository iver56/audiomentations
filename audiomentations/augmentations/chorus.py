import random

import numpy as np
from numpy.typing import NDArray

from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import convert_decibels_to_amplitude_ratio


class Chorus(BaseWaveformTransform):
    """
    Multiply the audio by a random amplitude factor to reduce or increase the volume. This
    technique can help a model become somewhat invariant to the overall gain of the input audio.

    Warning: This transform can return samples outside the [-1, 1] range, which may lead to
    clipping or wrap distortion, depending on what you do with the audio in a later stage.
    See also https://en.wikipedia.org/wiki/Clipping_(audio)#Digital_clipping
    """

    supports_multichannel = True

    def __init__(
        self,
        min_chorus_rate: float = 0.1,
        max_chorus_rate: float = 0.3,
        min_chorus_depth_ms: float = 10,
        max_chorus_depth_ms: float = 18,
        min_offset_ms: float = 20,
        max_offset_ms: float = 40,
        p: float = 0.5,
    ):
        """
        :param min_chorus_rate: Minimum chorus rate in Hz
        :param max_chorus_rate: Maximum chorus rate in Hz
        :param min_chorus_depth_ms: Minimum chorus depth in milliseconds
        :param max_chorus_depth_ms: Maximum chorus depth in milliseconds
        :param min_offset_ms: Minimum delay time offset in milliseconds
        :param max_offset_ms: Maximum delay time offset in milliseconds
        :param p: The probability of applying this transform
        """
        super().__init__(p)

        if min_chorus_rate > max_chorus_rate:
            raise ValueError("min_chorus_rate cannot be greater than max_chorus_rate")
            
        if min_chorus_depth_ms > max_chorus_depth_ms:
            raise ValueError("min_chorus_depth_ms cannot be greater than max_chorus_depth_ms")
            
        if min_offset_ms > max_offset_ms:
            raise ValueError("min_offset_ms cannot be greater than max_offset_ms")
            
        self.min_chorus_rate = min_chorus_rate
        self.max_chorus_rate = max_chorus_rate
        
        self.min_chorus_depth_ms = min_chorus_depth_ms
        self.max_chorus_depth_ms = max_chorus_depth_ms
        
        self.min_offset_ms = min_offset_ms
        self.max_offset_ms = max_offset_ms

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["chorus_rate"] = random.uniform(
                self.min_chorus_rate, self.max_chorus_rate
            )
            self.parameters["chorus_depth"] = random.uniform(
                self.min_chorus_depth_ms / 1000 * sample_rate , self.max_chorus_depth_ms / 1000 * sample_rate
            )
            self.parameters["offset"] = random.uniform(
                self.min_offset_ms / 1000 * sample_rate, self.max_offset_ms / 1000 * sample_rate
            )
            
    def variable_delay(self, samples, delay_times):
        delayed_samples = np.zeros_like(samples)
        for i, delay in enumerate(delay_times):
            if i >= delay:
                delayed_samples[i] = samples[i - delay]
        return delayed_samples

    def apply(self, samples: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
        n = samples.shape[-1]
        if len(samples.shape) > 1:
            distorted_samples = np.zeros((samples.shape[0], n), dtype=np.float32)
            for i in range(samples.shape[0]):
                lfo = 0.5 + 0.5 * np.sin(2 * np.pi * self.parameters["chorus_rate"] * np.arange(n) / sample_rate)
                delay_times = (lfo * self.parameters["chorus_depth"] + self.parameters["offset"]).astype(int)
                distorted_samples[i] = self.variable_delay(samples[i], delay_times)
        else:
            lfo = 0.5 + 0.5 * np.sin(2 * np.pi * self.parameters["chorus_rate"] * np.arange(n) / sample_rate)
            delay_times = (lfo * self.parameters["chorus_depth"] + self.parameters["offset"]).astype(int)
            distorted_samples = self.variable_delay(samples, delay_times)
        return distorted_samples
