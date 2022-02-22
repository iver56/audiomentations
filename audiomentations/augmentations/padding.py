import numpy as np

from audiomentations.core.transforms_interface import BaseWaveformTransform

class Padding(BaseWaveformTransform):
    """
    Apply padding to the audio signal -  take a fraction of the end or the start of the audio
    and treat that part as padding.
    """
    
    supports_multichannel = True
    
    def __init__(self, mode='constant', p=0.5):
        super().__init__(p)
        self.mode = mode

    def apply(self, samples, sample_rate):
        orig_len = len(samples)
        skip_idx = np.random.randint(1, orig_len-1)
        r = np.random.random()
        if r < 0.5:
            samples = samples[:skip_idx]
        else:
            samples = samples[-skip_idx:]
        pad_width = orig_len - len(samples)
        samples = np.pad(samples, pad_width, self.mode)
        if r < 0.5:
            samples = samples[:orig_len]
        else:
            samples = samples[-orig_len:]
        return samples
