import numpy as np
from audiomentations.core.transforms_interface import BaseWaveformTransform

class InterruptPulse(BaseWaveformTransform):
    """
    Add interrupt pulses to the input audio. This can be used to simulate the signals when the ECG electrodes are in lead-off condition.
    The function generates random number of flat pulses with some noise on it.
    The pulses' amplitudes are between signal max-min range and pulses have random width.
    """    
    support_multichannel = False
    
    def __init__(
        self,
        min_num_interruptions: int = 0,
        max_num_interruptions: int = 3,
        noise_level: float = 100.,
        p: float = 0.5
    ):
        """
        :param min_num_interruptions: Minimum number of interruptions
        :param max_num_interruptions: Maximum number of interruptions
        :noise_level: Noise level
        :param p: The probability of applying this transform
        """
        super().__init__(p)

        self.min_num_interruptions = min_num_interruptions
        self.max_num_interruptions = max_num_interruptions
        self.noise_level = noise_level
        if self.min_num_interruptions > self.max_num_interruptions:
            raise ValueError("min_num_interruptions must not be greater than max_num_interruptions")
            
    def randomize_parameters(
        self, samples: np.array, sample_rate: int = None
    ):
        super().randomize_parameters(samples, sample_rate)

        self.parameters["num_interruptions"] = np.random.randint(
            low=self.min_num_interruptions,
            high=self.max_num_interruptions
        )
        
    def apply(self, samples: np.array, sample_rate: int = None):
        max_peak = np.max(samples)
        min_peak = np.min(samples)
        num_samples = samples.shape[-1]
        
        interruption_start_times = num_samples * np.random.random(np.ceil(self.parameters["num_interruptions"]*4/5).astype(int)).astype(int)
        for interruption in interruption_start_times:
            interruption_len = int(num_samples*np.random.random()//5)
            interruption_val = np.random.uniform(min_peak, max_peak)
            samples[interruption:interruption+interruption_len] = interruption_val+np.random.normal(0, 1, interruption_len)*self.noise_level
        return samples
