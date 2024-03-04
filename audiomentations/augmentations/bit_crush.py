import random

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
    
    def __init__(self, min_bit_depth: int = 5, max_bit_depth: int = 10, p: float = 0.5):
        """
        :param bit_depth: The bit depth to which the audio sample will be converted to
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        self.min_bit_depth = min_bit_depth
        self.max_bit_depth = max_bit_depth
        
        assert self.min_bit_depth <= self.max_bit_depth
        
    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["bit_depth"] = random.uniform(
                self.min_bit_depth, self.max_bit_depth
            )
        
    def apply(self, samples: NDArray[np.float32], sample_rate: int):
        q = (2**self.parameters["bit_depth"]/2) + 1
        return np.round(samples * q)/q
        