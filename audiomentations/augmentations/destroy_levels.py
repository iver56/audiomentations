import numpy as np

from audiomentations.core.transforms_interface import BaseWaveformTransform

class DestroyLevels(BaseWaveformTransform):
    supports_multichannel = True

    def __init__(self,
                 min_n_levels=2,
                 max_n_levels=10,
                 min_level=0.3,
                 max_level=1,
                 p=0.5):
        super().__init__(p)
        self.min_n_levels = min_n_levels
        self.max_n_levels = max_n_levels
        self.min_level = min_level
        self.max_level = max_level
        assert self.min_n_levels <= self.max_n_levels
        assert self.min_level <= self.max_level

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            n_levels = np.random.randint(
                self.min_n_levels, self.max_n_levels
            )
            
            levels = np.random.uniform(
                self.min_level, self.max_level, n_levels
            )
            
            starts = [0, *sorted(np.random.randint(0, samples.shape[-1], n_levels - 1))]
            
            self.parameters['levels'] = list(zip(starts, levels))

            
    def apply(self, samples, sample_rate):
        f = np.ones((samples.shape[-1]))
        for s, l in self.parameters['levels']:
            f[s:] = l

        compressed = samples * f        
        return compressed