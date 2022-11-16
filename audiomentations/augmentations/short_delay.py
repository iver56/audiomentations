import random
import numpy as np

from audiomentations.core.utils import random_log_int, apply_ffmpeg_commands
from audiomentations.core.transforms_interface import BaseWaveformTransform

class ShortDelay(BaseWaveformTransform):
    supports_multichannel = True

    def __init__(self,
                 min_gain=0.5,
                 max_gain=1,
                 min_delay=0,
                 max_delay=50,
                 p=0.5):
        super().__init__(p)
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.min_delay = min_delay
        self.max_delay = max_delay
        assert self.min_gain <= self.max_gain
        assert self.min_delay <= self.max_delay

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters['gain'] = random.uniform(
                self.min_gain, self.max_gain
            )
            
            self.parameters['delay'] = random.uniform(
                self.min_delay, self.max_delay
            )

            
    def apply(self, samples, sample_rate):
        delay = self.parameters['delay']
        num_samples = round(delay / 1000 * sample_rate)
        
        delayed_samples = np.concatenate([np.zeros(num_samples), samples[:-num_samples]])
        delayed_samples = delayed_samples * self.parameters['gain']

        assert delayed_samples.shape == samples.shape

        return delayed_samples * 0.5 + samples * 0.5