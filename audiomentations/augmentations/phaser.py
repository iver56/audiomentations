import random

from audiomentations.core.utils import apply_ffmpeg_commands
from audiomentations.core.transforms_interface import BaseWaveformTransform

class Phaser(BaseWaveformTransform):
    supports_multichannel = True

    def __init__(self,
                 min_gain=0.5,
                 max_gain=1,
                 min_speed=0.1,
                 max_speed=2,
                 modulation_types=['sinusoidal', 'triangular'],
                 p=0.5):
        super().__init__(p)
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.modulation_types = modulation_types
        assert self.min_gain <= self.max_gain
        assert self.min_speed <= self.max_speed
        for mt in self.modulation_types:
            assert mt in ['sinusoidal', 'triangular']

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters['gain'] = random.uniform(
                self.min_gain, self.max_gain
            )
            
            self.parameters['speed'] = random.uniform(
                self.min_speed, self.max_speed
            )

            self.parameters['modulation_type'] = random.choice(self.modulation_types)

            
    def apply(self, samples, sample_rate):
        ffmpeg_command = "aphaser=out_gain={}:speed={}:type={}".format(
            self.parameters['gain'],
            self.parameters['speed'],
            self.parameters['modulation_type']
        )
        
        compressed = apply_ffmpeg_commands(samples, sample_rate, ['-af', ffmpeg_command])

        assert compressed.shape == samples.shape
        
        return compressed