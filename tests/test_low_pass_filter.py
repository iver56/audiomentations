import unittest
import scipy
import scipy.signal

import numpy as np

import matplotlib.pyplot as plt

from audiomentations import LowPassFilter

DEBUG = False


class TestLowPassFilter(unittest.TestCase):
    def test_low_pass_filter(self):

        np.random.seed(1)

        sample_rate = 8000

        # Filter cutoff
        fc_lowpass = 3000

        # Parameters for computing periodograms
        nfft = 1024
        nperseg = 1024

        # Create a 20 seconds chirp from 50Hz to 250Hz
        n = np.arange(0, 20 , 1/sample_rate)
        samples = scipy.signal.chirp(n, 50, 20, 250, method='linear')

        augment = LowPassFilter(min_cutoff_freq=100, max_cutoff_freq=200, p=1.0)
        processed_samples = augment(samples=samples, sample_rate=sample_rate)
        fc = augment.parameters['cutoff_freq']

        # Compute periodograms
        wx, samples_pxx = scipy.signal.welch(samples, fs=sample_rate, nfft=nfft, nperseg=nperseg, scaling='spectrum',  window='hann')
        _, processed_samples_pxx = scipy.signal.welch(processed_samples, fs=sample_rate, nperseg=nperseg, nfft=nfft, scaling='spectrum', window='hann')

        # Compute db at cutoffs between the ideal
        samples_db_at_fc = 10*np.log10(samples_pxx[int(np.round(nfft/sample_rate * fc))])
        processed_samples_db_at_fc = 10*np.log10(processed_samples_pxx[int(np.round(nfft/sample_rate * fc))])


        if DEBUG:
            plt.plot(wx, 10*np.log10(np.abs(samples_pxx)))
            plt.plot(wx, 10*np.log10(np.abs(processed_samples_pxx)), ':')
            plt.legend(['Input signal', f'Highpassed at f_c={fc:.2f}'])
            plt.axvline(fc, color='red', linestyle=':')
            plt.axhline(samples_db_at_fc-3, color='red',  linestyle=':')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude (dB)')
            plt.show()

        self.assertEqual(processed_samples.shape, samples.shape)
        self.assertEqual(processed_samples.dtype, np.float32)
        
        # Assert that at fc we are at the 3db point give or take half a db.
        self.assertTrue(np.isclose(samples_db_at_fc - processed_samples_db_at_fc, 3, 0.5))
