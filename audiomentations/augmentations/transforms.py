import numpy as np

from audiomentations.core.transforms_interface import BasicTransform


class AddGaussianNoise(BasicTransform):
    def apply(self, samples, sample_rate):
        noise = np.random.randn(len(samples))
        samples = samples + 0.005 * noise
        return samples
