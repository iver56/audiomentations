import random

from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import apply_ffmpeg_commands, random_log_int

class Compressor(BaseWaveformTransform):
    supports_multichannel = True

    def __init__(self,
                 min_ratio=1,
                 max_ratio=20,
                 min_threshold=-50,
                 max_threshold=-10,
                 min_attack=0.01,
                 max_attack=2000,
                 min_release=0.01,
                 max_release=9000,
                 min_makeup=1,
                 max_makeup=64,
                 min_knee=1,
                 max_knee=8,
                 p=0.5):
        super().__init__(p)
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.min_attack = min_attack
        self.max_attack = max_attack
        self.min_release = min_release
        self.max_release = max_release
        self.min_makeup = min_makeup
        self.max_makeup = max_makeup
        self.min_knee = min_knee
        self.max_knee = max_knee
        assert self.min_ratio <= self.max_ratio
        assert self.min_threshold <= self.max_threshold
        assert self.min_ratio <= self.max_ratio
        assert self.min_attack <= self.max_attack
        assert self.min_release <= self.max_release
        assert self.min_makeup <= self.max_makeup
        assert self.min_knee <= self.max_knee

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters['ratio'] = random.uniform(
                self.min_ratio, self.max_ratio
            )
            
            self.parameters['threshold'] = random.randint(
                self.min_threshold, self.max_threshold
            )
            
            self.parameters['attack'] = random.uniform(
                self.min_attack, self.max_attack
            )
            
            self.parameters['release'] = random.uniform(
                self.min_release, self.max_release
            )
            
            self.parameters['makeup'] = random_log_int(
                self.min_makeup, self.max_makeup
            )
            
            self.parameters['knee'] = random.uniform(
                self.min_knee, self.max_knee
            )

    def apply(self, samples, sample_rate):
        ffmpeg_command = "acompressor=threshold={}dB:ratio={}:attack={}:release={}:makeup={}:knee={}"
        ffmpeg_command = ffmpeg_command.format(
            self.parameters['threshold'],
            self.parameters['ratio'],
            self.parameters['attack'],
            self.parameters['release'],
            self.parameters['makeup'],
            self.parameters['knee']
        )

        compressed = apply_ffmpeg_commands(samples, sample_rate, ['-af', ffmpeg_command])

        assert compressed.shape == samples.shape

        return compressed