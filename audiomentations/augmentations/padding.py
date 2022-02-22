import numpy as np

from audiomentations.core.transforms_interface import BaseWaveformTransform

class Padding(BaseWaveformTransform):
    """
    Apply padding to the audio signal -  take a fraction of the end or the start of the audio
    and treat that part as padding.
    """
    
    supports_multichannel = True
    
    def __init__(self, mode='constant', p=0.5):
        """
        :param mode: Padding mode. Must be one of 'constant', 'edge', 'wrap', 'reflect'
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        self.mode = mode

    def apply(self, samples, sample_rate):
        orig_len = samples.shape[-1]
        if len(samples.shape) > 1:
            n_channels = samples.shape[0]
        else:
            n_channels = 1
        skip_idx = np.random.randint(1, orig_len-1)
        r = np.random.random()
        if r < 0.5:
            samples = samples[..., :skip_idx]
        else:
            samples = samples[..., -skip_idx:]
            
        if n_channels > 1:
            pad_width =  ((0, 0)*(n_channels-1), (orig_len, orig_len - samples.shape[-1]))
        else:
            pad_width = orig_len, orig_len - samples.shape[-1]
        samples = np.pad(samples, pad_width, self.mode)
        if r < 0.5:
            samples = samples[..., :orig_len]
        else:
            samples = samples[..., -orig_len:]
        return samples
