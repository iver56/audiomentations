import numpy as np

from audiomentations.core.transforms_interface import BaseWaveformTransform

class Padding(BaseWaveformTransform):
    """
    Apply padding to the audio signal -  take a fraction of the end or the start of the audio
    and treat that part as padding.
    """
    
    supports_multichannel = True
    
    def __init__(self, pad_width, mode='constant', p=0.5):
        super().__init__(p)
        self.pad_width = pad_width
        self.mode = mode

    def apply(self, samples, sample_rate):
        samples = np.pad(samples, self.pad_width, self.mode)
        return samples
