import random
import numpy as np

from audiomentations.core.transforms_interface import BaseWaveformTransform

class GainPeak(BaseWaveformTransform):
    """
    Apply a gain peak, i.e turn the gain up, and then down.
    """
    
    supports_multichannel = True
    
    def __init__(self, min_gain, min_gain_diff, max_gain_diff, p=0.5):
        """
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        self.min_gain = min_gain
        self.min_gain_diff = min_gain_diff
        self.max_gain_diff = max_gain_diff
        
    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["gain_diff"] = random.uniform(self.min_gain_diff, self.max_gain_diff)
        
    def apply(self, samples, sample_rate):
        n_samples_half = len(samples)//2
        gain_up = np.linspace(self.min_gain, self.min_gain+self.parameters["gain_diff"], 
                        num=n_samples_half, dtype=samples.dtype)
        gain_down = np.linspace(self.min_gain+self.parameters["gain_diff"], self.min_gain,
                        num=len(samples) - n_samples_half, dtype=samples.dtype)
        gain = np.concatenate((gain_up, gain_down))
                        
        if samples.ndim > 1:
            if samples.shape[1] > 1:
                gain = np.repeat(gain[:, np.newaxis], samples.shape[1], axis=1)

        return samples*gain
        
        