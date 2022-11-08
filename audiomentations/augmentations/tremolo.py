import random

from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import apply_ffmpeg_commands

class Tremolo(BaseWaveformTransform):
    supports_multichannel = True

    def __init__(self,
                 min_f=-.1,
                 max_f=10,
                 min_d=0.3,
                 max_d=1,
                 p=0.5):
        super().__init__(p)
        self.min_f = min_f
        self.max_f = max_f
        self.min_d = min_d
        self.max_d = max_d
        assert self.min_f <= self.max_f
        assert self.min_d <= self.max_d

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters['f'] = random.uniform(
                self.min_f, self.max_f
            )
            
            self.parameters['d'] = random.uniform(
                self.min_d, self.max_d
            )

    def apply(self, samples, sample_rate):
        
        # Ref: https://ffmpeg.org/ffmpeg-all.html#tremolo
        ffmpeg_command = f"tremolo=f={self.parameters['f']}:d={self.parameters['d']}"

        compressed = apply_ffmpeg_commands(samples, sample_rate, ['-af', ffmpeg_command])

        assert compressed.shape == samples.shape
        
        return compressed