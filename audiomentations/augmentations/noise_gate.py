import random

from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import apply_ffmpeg_commands

class NoiseGate(BaseWaveformTransform):
    supports_multichannel = True

    def __init__(self,
                 min_threshold=-60,
                 max_threshold=-35,
                 p=0.5):
        super().__init__(p)
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        assert self.min_threshold <= self.max_threshold

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters['threshold'] = random.uniform(
                self.min_threshold, self.max_threshold
            )

    def apply(self, samples, sample_rate):
        threshold = self.parameters["threshold"]
        next_db = threshold + 0.1
        
        # Ref: https://ffmpeg.org/ffmpeg-filters.html#Examples-22
        ffmpeg_command = f"compand=.1:.2:-900/-900|{threshold}/-900|{next_db}/{next_db}:.01:0:-90:.1"

        compressed = apply_ffmpeg_commands(samples, sample_rate, ['-af', ffmpeg_command])

        assert compressed.shape == samples.shape
        
        return compressed