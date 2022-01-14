from audiomentations.augmentations.transforms import LowPassFilter, HighPassFilter

import pytest
import scipy
import scipy.signal

import numpy as np




DEBUG = False

def get_chirp_test(sample_rate, duration):
    """
    Create a `duration` seconds chirp from 2Hz to `nyquist frequency` - 10 Hz
    """

    n = np.arange(0, duration, 1 / sample_rate)
    samples = scipy.signal.chirp(n, 2, duration, sample_rate // 2 - 10, method="linear")
    return samples


class TestOneSidedFilterTransforms:

    @pytest.mark.parametrize("filter_type", ["lowpass", "highpass"])
    @pytest.mark.parametrize(
        "rolloff",
        [6, 12, 18, 120],
    )
    def test_one_single_input(self, filter_type, rolloff):

        np.random.seed(1)

        sample_rate = 8000

        samples = get_chirp_test(sample_rate, 40)

        # Parameters for computing periodograms
        nfft = 1024
        nperseg = 1024

        if filter_type == "highpass":
            FilterTransform = HighPassFilter
        elif filter_type == "lowpass":
            FilterTransform = LowPassFilter

        augment = FilterTransform(
            min_cutoff_freq=100,
            # max_cutoff_freq must be less than half nyquist frequency
            # for the test at double fc below to work.
            max_cutoff_freq=1900,
            min_rolloff=rolloff,
            max_rolloff=rolloff,
            p=1.0,
        )
        processed_samples = augment(samples=samples, sample_rate=sample_rate)
        fc = augment.parameters["cutoff_freq"]

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
        samples_db_at_fc = 10 * np.log10(
            samples_pxx[int(np.round(nfft / sample_rate * fc))]
        )

        samples_db_below_fc = 10 * np.log10(
            samples_pxx[int(np.round(nfft / sample_rate * fc * 2))]
        )
        samples_db_above_fc = 10 * np.log10(
            samples_pxx[int(np.round(nfft / sample_rate * fc * 2))]
        )

        processed_sample_db_at_fc = 10 * np.log10(
            processed_samples_pxx[int(np.round(nfft / sample_rate * fc))]
        )

        processed_sample_db_below_fc = 10 * np.log10(
            processed_samples_pxx[int(np.round(nfft / sample_rate * fc / 2))]
        )

        processed_sample_db_above_fc = 10 * np.log10(
            processed_samples_pxx[int(np.round(nfft / sample_rate * fc * 2))]
        )

        if DEBUG:
            import matplotlib.pyplot as plt
            plt.plot(wx, 10 * np.log10(np.abs(samples_pxx)))
            plt.plot(wx, 10 * np.log10(np.abs(processed_samples_pxx)), ":")
            plt.legend(["Input signal", f"Highpassed at f_c={fc:.2f}"])
            plt.axvline(fc, color="red", linestyle=":")
            plt.axhline(samples_db_at_fc - 3, color="red", linestyle=":")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude (dB)")
            plt.show()

        assert processed_samples.shape == samples.shape
        assert processed_samples.dtype == np.float32

        # Assert that at fc we are at the 3db point give or take half a db.
        assert np.isclose(samples_db_at_fc - processed_sample_db_at_fc, 3, 0.5)

        if filter_type == "highpass":
            # Assert the point at half fc (stopband) is around 3 + <rolloff> dB
            assert np.isclose(
                samples_db_below_fc - processed_sample_db_below_fc, 3 + rolloff, 1
            )

            # Assert the point at double fc (passband) is around 0dB
            assert np.isclose(samples_db_above_fc, processed_sample_db_above_fc, 0.5)

        elif filter_type == "lowpass":
            # Assert the point at half fc (stopband) is around 3 + <rolloff> dB
            assert np.isclose(
                samples_db_above_fc - processed_sample_db_above_fc, 3 + rolloff, 1
            )

            # Assert the point at double fc (passband) is around 0dB
            assert np.isclose(samples_db_below_fc, processed_sample_db_below_fc, 0.5)

    @pytest.mark.parametrize(
        "samples",
        [get_chirp_test(8000, 40)],
    )
    @pytest.mark.parametrize("filter_type", ["lowpass", "highpass"])
    def test_two_channel_input(self, samples, filter_type):

        sample_rate = 8000
        samples = get_chirp_test(sample_rate, 40)

        # Convert to 2D two channels
        two_channels = np.vstack([samples, samples])

        if filter_type == "highpass":
            FilterTransform = HighPassFilter
        elif filter_type == "lowpass":
            FilterTransform = LowPassFilter

        augment = FilterTransform(
            min_cutoff_freq=100,
            max_cutoff_freq=100,
            min_rolloff=6,
            max_rolloff=6,
            p=1.0,
        )

        processed_samples = augment(samples=two_channels, sample_rate=sample_rate)

        processed_two_channels = augment(samples=two_channels, sample_rate=sample_rate)

        assert processed_two_channels.shape[0] == 2
        assert processed_two_channels.shape == two_channels.shape
        assert processed_two_channels.dtype == np.float32

        # Check that the processed 2D channel version applies the same effect
        # as the passband version.
        for n, channel in enumerate(processed_two_channels):
            assert np.allclose(channel, processed_samples[n])
