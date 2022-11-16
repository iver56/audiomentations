import numpy as np

from audiomentations.core.utils import random_log_int, apply_ffmpeg_commands
from audiomentations.core.transforms_interface import BaseWaveformTransform

class TwoPoleAllPassFilter(BaseWaveformTransform):
    supports_multichannel = True

    def __init__(self,
                 min_frequency=1,
                 max_frequency=8000,
                 min_blend=0,
                 max_blemd=1,
                 p=0.5):
        super().__init__(p)
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.min_blend = min_blend
        self.max_blemd = max_blemd
        assert self.min_frequency <= self.max_frequency
        assert self.min_blend <= self.max_blemd

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters['frequency'] = random_log_int(
                self.min_frequency, self.max_frequency
            )
            
            blend = np.random.normal(0.5, 0.125)

            self.parameters['blend'] = min(self.max_blemd, max(self.min_blend, blend))

            
    def apply(self, samples, sample_rate):
        ffmpeg_command = f"allpass=frequency={self.parameters['frequency']}"
        
        compressed = apply_ffmpeg_commands(samples, sample_rate, ['-af', ffmpeg_command])
        assert compressed.shape == samples.shape
        
        b = self.parameters['blend']
        return compressed * b + samples * (1 - b)