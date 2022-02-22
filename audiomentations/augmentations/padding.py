import numpy as np

from audiomentations.core.transforms_interface import BaseWaveformTransform

class Padding(BaseWaveformTransform):
    """
    Apply padding to the audio signal -  take a fraction of the end or the start of the audio
    and treat that part as padding.
    """
    
    supports_multichannel = True
    
    def __init__(self, mode='constant', min_fraction=0.1, max_fraction=0.5, p=0.5):
        """
        :param mode: Padding mode. Must be one of 'constant', 'edge', 'wrap', 'reflect'
        :param min_fraction: Minimum part of signal to be padded
        :param max_fraction: Maximum part of signal to be padded
        :param p: The probability of applying this transform
        """
        super().__init__(p)
    
        assert mode == 'constant' or mode == 'edge' or mode == 'wrap' \
                or mode == 'reflect'
        assert min_fraction < 1. and max_fraction < 1.
        assert min_fraction > 0 and max_fraction > 0    
        assert min_fraction < max_fraction
         
        self.mode = mode
        self.min_fraction = min_fraction
        self.max_fraction = max_fraction

    def apply(self, samples, sample_rate):
        orig_len = samples.shape[-1]
        if len(samples.shape) > 1:
            n_channels = samples.shape[0]
        else:
            n_channels = 1
        
        a = int(np.ceil(self.min_fraction*orig_len))
        b = int(np.floor(self.max_fraction*orig_len))
        
        skip_idx = np.random.randint(a, b)
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
