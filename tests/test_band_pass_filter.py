import unittest

import numpy as np

import matplotlib.pyplot as plt

from audiomentations.augmentations.transforms import BandPassFilter

DEBUG = False

class TestBandPassFilter(unittest.TestCase):
    def test_band_pass_filter(self):
        sample_rate = 16000
        t = .25                # signal duration in sec
        f = 500                # signal frequency in Hz
        samples = np.arange(t * f, dtype=np.float32) / sample_rate
        samples = np.sin(2 * np.pi * f * samples)

        augment = BandPassFilter(min_center_freq=100, 
                                 max_center_freq=5000,
                                 min_q=1.0,
                                 max_q=2.0,
                                 p=1.0)
        processed_samples = augment(
            samples=samples, sample_rate=sample_rate
        )

        self.assertEqual(processed_samples.shape, samples.shape)
        self.assertEqual(processed_samples.dtype, np.float32)
        
        if DEBUG:        
            plt.plot(samples)
            plt.plot(processed_samples, '-.')
            plt.show()
