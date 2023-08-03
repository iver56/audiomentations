import random

from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import apply_ffmpeg_commands

class SimpleCompressor(BaseWaveformTransform):
    supports_multichannel = True

    def __init__(self,
                 min_ratio=1,
                 max_ratio=20,
                 p=0.5):
        super().__init__(p)
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        assert self.min_ratio <= self.max_ratio

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters['ratio'] = random.uniform(
                self.min_ratio, self.max_ratio
            )

    def apply(self, samples, sample_rate):
        ffmpeg_command = f"acompressor=ratio={self.parameters['ratio']}"

        compressed = apply_ffmpeg_commands(samples, sample_rate, ['-af', ffmpeg_command])

        assert compressed.shape == samples.shape

        return compressed