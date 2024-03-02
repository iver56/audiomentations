import numpy as np
from numpy.typing import NDArray

from audiomentations.core.transforms_interface import BaseWaveformTransform

class BitCrush(BaseWaveformTransform):
    """
    Apply a bit crush effect the audio by reducing of the resolution or bandwidth of audio 
    data. This class implements bit depth reduction, which reduces the number of bits 
    that can be used for representing each audio sample.
    """
    
    supports_multichannel = True
    
    def __init__(self, bit_depth: int = 4, p: float = 0.5):
        """
        :param bit_depth: The bit depth to which the audio sample will be converted to
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        self.bit_depth = bit_depth
        
    def apply(self, samples: NDArray[np.float32], sample_rate: int):
        q = (2**self.bit_depth/2) + 1
        return np.round(samples * q)/q
        