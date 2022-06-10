import random
import numpy as np

from audiomentations.core.transforms_interface import BaseWaveformTransform

class GainPeak(BaseWaveformTransform):
    """
    Apply a gain peak, i.e turn the gain up, and then down.
    """
    
    supports_multichannel = True
    
    def __init__(self, min_gain=1.0, min_gain_diff=0.1, max_gain_diff=0.5, 
            min_peak_relpos=0.3, max_peak_relpos=0.7, p=0.5):
        """
        :param min_gain: minimal gain
        :param min_gain_diff: minimal difference between min and max gain
        :param max_gain_diff: maximal difference between min and max gain
        :param min_peak_relpos: minimal peak relative position
        :param max_peak_relpos: maximal peak relative position
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        self.min_gain = min_gain
        self.min_gain_diff = min_gain_diff
        self.max_gain_diff = max_gain_diff
        self.min_peak_relpos = min_peak_relpos
        self.max_peak_relpos = max_peak_relpos
        
    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["gain_diff"] = random.uniform(self.min_gain_diff, self.max_gain_diff)
            self.parameters["peak_relpos"] = random.uniform(self.min_peak_relpos, self.max_peak_relpos)
        
    def apply(self, samples, sample_rate):
        peak_pos = int(len(samples)*self.parameters["peak_relpos"])
        gain_up = np.linspace(self.min_gain, self.min_gain+self.parameters["gain_diff"], 
                        num=peak_pos, dtype=samples.dtype)
        gain_down = np.linspace(self.min_gain+self.parameters["gain_diff"], self.min_gain,
                        num=len(samples) - peak_pos, dtype=samples.dtype)
        gain = np.concatenate((gain_up, gain_down))
                        
        if samples.ndim > 1:
            if samples.shape[1] > 1:
                gain = np.repeat(gain[:, np.newaxis], samples.shape[1], axis=1)

        return samples*gain
        
        