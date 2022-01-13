import unittest

import numpy as np
import scipy
import scipy.signal
import matplotlib.pyplot as plt

from audiomentations import BandStopFilter

DEBUG = False


class TestBandPassFilter(unittest.TestCase):
    def test_single_channel_input(self):

        np.random.seed(1)
        sample_rate = 8000

        # Parameters for computing periodograms
        nfft = 1024
        nperseg = 1024

        # Create a 20 seconds chirp from 50Hz to 250Hz
        n = np.arange(0, 20, 1 / sample_rate)
        samples = scipy.signal.chirp(n, 50, 20, 2000, method="linear")

        augment = BandStopFilter(
            min_center_freq=100.0, max_center_freq=1000.0, min_q=1.0, max_q=2.0, p=1.0
        )
        processed_samples = augment(samples=samples, sample_rate=sample_rate)
        fcL = augment.parameters["center_freq"] * (1 - 0.5 / augment.parameters["q"])
        fcH = augment.parameters["center_freq"] * (1 + 0.5 / augment.parameters["q"])

        # Compute periodograms
        wx, samples_pxx = scipy.signal.welch(
            samples,
            fs=sample_rate,
            nfft=nfft,
            nperseg=nperseg,
            scaling="spectrum",
            window="hann",
        )
        _, processed_samples_pxx = scipy.signal.welch(
            processed_samples,
            fs=sample_rate,
            nperseg=nperseg,
            nfft=nfft,
            scaling="spectrum",
            window="hann",
        )

        # Compute db at cutoffs at the input as well as the filtered signals
        samples_db_at_fcL = 10 * np.log10(
            samples_pxx[int(np.round(nfft / sample_rate * fcL))]
        )
        processed_samples_db_at_fcL = 10 * np.log10(
            processed_samples_pxx[int(np.round(nfft / sample_rate * fcL))]
        )

        samples_db_at_fcH = 10 * np.log10(
            samples_pxx[int(np.round(nfft / sample_rate * fcH))]
        )
        processed_samples_db_at_fcH = 10 * np.log10(
            processed_samples_pxx[int(np.round(nfft / sample_rate * fcH))]
        )

        if DEBUG:
            plt.figure(figsize=(11, 11))
            plt.plot(wx, 10 * np.log10(np.abs(samples_pxx)))
            plt.plot(wx, 10 * np.log10(np.abs(processed_samples_pxx)), ":")
            plt.legend(["Input signal", f"Bandpassed at f_l={fcL:.2f}, f_h={fcH:.2f}"])
            plt.axvline(fcL, color="red", linestyle=":")
            plt.axvline(fcH, color="red", linestyle=":")
            plt.axhline(samples_db_at_fcL - 3, color="red", linestyle=":")
            plt.axhline(samples_db_at_fcH - 3, color="red", linestyle="-.")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude (dB)")
            plt.show()

        self.assertEqual(processed_samples.shape, samples.shape)
        self.assertEqual(processed_samples.dtype, np.float32)

        # Assert that at fc we are at the 3db point give or take half a db.
        self.assertTrue(
            np.isclose(samples_db_at_fcL - processed_samples_db_at_fcL, 3, 0.5)
        )
        self.assertTrue(
            np.isclose(samples_db_at_fcH - processed_samples_db_at_fcH, 3, 0.5)
        )

    def test_two_channel_input(self):

        np.random.seed(1)
        sample_rate = 8000

        # Parameters for computing periodograms
        nfft = 1024
        nperseg = 1024

        # Create a 20 seconds chirp from 50Hz to 250Hz
        n = np.arange(0, 20, 1 / sample_rate)
        samples = scipy.signal.chirp(n, 50, 20, 2000, method="linear")

        # Convert to 2D two channels
        samples = np.vstack([samples, samples[::-1]])

        augment = BandStopFilter(
            min_center_freq=100.0, max_center_freq=1000.0, min_q=1.0, max_q=2.0, p=1.0
        )
        processed_samples = augment(samples=samples, sample_rate=sample_rate)

        self.assertEqual(processed_samples.shape[0], 2)
        self.assertEqual(processed_samples.shape, samples.shape)
        self.assertEqual(processed_samples.dtype, np.float32)

        for n, channel in enumerate(samples):

            processed_channel = processed_samples[n]

            fcL = augment.parameters["center_freq"] * (
                1 - 0.5 / augment.parameters["q"]
            )
            fcH = augment.parameters["center_freq"] * (
                1 + 0.5 / augment.parameters["q"]
            )

            # Compute periodograms
            wx, samples_pxx = scipy.signal.welch(
                channel,
                fs=sample_rate,
                nfft=nfft,
                nperseg=nperseg,
                scaling="spectrum",
                window="hann",
            )
            _, processed_samples_pxx = scipy.signal.welch(
                processed_channel,
                fs=sample_rate,
                nperseg=nperseg,
                nfft=nfft,
                scaling="spectrum",
                window="hann",
            )

            # Compute db at cutoffs at the input as well as the filtered signals
            samples_db_at_fcL = 10 * np.log10(
                samples_pxx[int(np.round(nfft / sample_rate * fcL))]
            )
            processed_samples_db_at_fcL = 10 * np.log10(
                processed_samples_pxx[int(np.round(nfft / sample_rate * fcL))]
            )

            samples_db_at_fcH = 10 * np.log10(
                samples_pxx[int(np.round(nfft / sample_rate * fcH))]
            )
            processed_samples_db_at_fcH = 10 * np.log10(
                processed_samples_pxx[int(np.round(nfft / sample_rate * fcH))]
            )

            if DEBUG:
                plt.figure(figsize=(11, 11))
                plt.title(f"Channel {n}")
                plt.plot(wx, 10 * np.log10(np.abs(samples_pxx)))
                plt.plot(wx, 10 * np.log10(np.abs(processed_samples_pxx)), ":")
                plt.legend(
                    ["Input signal", f"Bandstopped at f_l={fcL:.2f}, f_h={fcH:.2f}"]
                )
                plt.axvline(fcL, color="red", linestyle=":")
                plt.axvline(fcH, color="red", linestyle=":")
                plt.axhline(samples_db_at_fcL - 3, color="red", linestyle=":")
                plt.axhline(samples_db_at_fcH - 3, color="red", linestyle="-.")
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("Magnitude (dB)")
                plt.show()

            # Assert that at fc we are at the 3db point give or take half a db.
            self.assertTrue(
                np.isclose(samples_db_at_fcL - processed_samples_db_at_fcL, 3, 0.5)
            )
            self.assertTrue(
                np.isclose(samples_db_at_fcH - processed_samples_db_at_fcH, 3, 0.5)
            )
