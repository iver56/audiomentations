import numpy as np
import pytest
import scipy
import scipy.signal

from audiomentations.augmentations.transforms import (
    BandPassFilter,
    BandStopFilter,
    HighPassFilter,
    LowPassFilter,
)

DEBUG = False


def get_chirp_test(sample_rate, duration):
    """
    Create a `duration` seconds chirp from 0Hz to `nyquist frequency`
    """

    n = np.arange(0, duration, 1 / sample_rate)
    samples = scipy.signal.chirp(n, 0, duration, sample_rate // 2, method="linear")
    return samples


def get_randn_test(sample_rate, duration):
    """
    Create a random noise test stimulus
    """

    n_samples = int(duration * sample_rate)

    samples = np.random.randn(n_samples)

    return samples


class TestOneSidedFilterTransforms:
    @pytest.mark.parametrize("filter_type", ["lowpass", "highpass"])
    @pytest.mark.parametrize(
        "rolloff",
        [6, 12, 18, 120],
    )
    @pytest.mark.parametrize("zero_phase", [False, True])
    def test_one_single_input(self, filter_type, rolloff, zero_phase):

        np.random.seed(1)

        sample_rate = 8000

        samples = get_chirp_test(sample_rate, 40)

        # Parameters for computing periodograms
        nfft = 1024
        nperseg = 1024

        # Expected db drop at fc
        if zero_phase:
            expected_db_drop = 6
        else:
            expected_db_drop = 3

        if filter_type == "highpass":
            FilterTransform = HighPassFilter
        elif filter_type == "lowpass":
            FilterTransform = LowPassFilter

        if zero_phase and rolloff % 12 != 0:
            with pytest.raises(AssertionError):
                augment = FilterTransform(
                    min_cutoff_freq=100,
                    # max_cutoff_freq must be less than half nyquist frequency
                    # for the test at double fc below to work.
                    max_cutoff_freq=1900,
                    min_rolloff=rolloff,
                    max_rolloff=rolloff,
                    zero_phase=zero_phase,
                    p=1.0,
                )
            return
        else:
            augment = FilterTransform(
                min_cutoff_freq=100,
                # max_cutoff_freq must be less than half nyquist frequency
                # for the test at double fc below to work.
                max_cutoff_freq=1900,
                min_rolloff=rolloff,
                max_rolloff=rolloff,
                zero_phase=zero_phase,
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

            plt.title(
                f"Filter type:{filter_type} Roll-off:{rolloff}db/octave Zero-phase:{zero_phase}"
            )
            plt.plot(wx, 10 * np.log10(np.abs(samples_pxx)))
            plt.plot(wx, 10 * np.log10(np.abs(processed_samples_pxx)), ":")
            plt.legend(["Input signal", f"Highpassed at f_c={fc:.2f}"])
            plt.axvline(fc, color="red", linestyle=":")
            plt.axhline(
                samples_db_at_fc - (3 if not zero_phase else 6),
                color="red",
                linestyle=":",
            )
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude (dB)")
            plt.show()

        assert processed_samples.shape == samples.shape
        assert processed_samples.dtype == np.float32

        # Assert that at fc we are at the 3db (6dB at the zero-phase case) point give or take half a db.
        assert np.isclose(
            samples_db_at_fc - expected_db_drop,
            processed_sample_db_at_fc,
            0.5,
        )

        if filter_type == "highpass":
            # Assert the point at half fc (stopband) is around 3 (6dB at the zero-phase case) + <rolloff> dB
            assert np.isclose(
                samples_db_below_fc - processed_sample_db_below_fc,
                expected_db_drop + rolloff,
                1,
            )

            # Assert the point at double fc (passband) is around 0dB
            assert np.isclose(samples_db_above_fc, processed_sample_db_above_fc, 0.5)

        elif filter_type == "lowpass":
            # Assert the point at half fc (stopband) is around 3 (or 6) + <rolloff> dB
            assert np.isclose(
                samples_db_above_fc - processed_sample_db_above_fc,
                expected_db_drop + rolloff,
                1,
            )

            # Assert the point at double fc (passband) is around 0dB
            assert np.isclose(samples_db_below_fc, processed_sample_db_below_fc, 0.5)

    @pytest.mark.parametrize(
        "samples",
        [get_chirp_test(8000, 40)],
    )
    @pytest.mark.parametrize("filter_type", ["lowpass", "highpass"])
    @pytest.mark.parametrize(
        "rolloff",
        [12, 120],
    )
    @pytest.mark.parametrize("zero_phase", [False, True])
    def test_two_channel_input(self, samples, filter_type, rolloff, zero_phase):

        sample_rate = 8000
        samples = get_randn_test(sample_rate, 10)

        # Convert to 2D two channels
        two_channels = np.vstack([samples, samples])

        if filter_type == "highpass":
            FilterTransform = HighPassFilter
        elif filter_type == "lowpass":
            FilterTransform = LowPassFilter

        augment = FilterTransform(
            min_cutoff_freq=1000,
            max_cutoff_freq=1000,
            min_rolloff=rolloff,
            max_rolloff=rolloff,
            zero_phase=zero_phase,
            p=1.0,
        )

        processed_samples = augment(samples=samples, sample_rate=sample_rate)

        processed_two_channels = augment(samples=two_channels, sample_rate=sample_rate)

        assert processed_two_channels.shape[0] == 2
        assert processed_two_channels.shape == two_channels.shape
        assert processed_two_channels.dtype == np.float32

        # Check that the processed 2D channel version applies the same effect
        # as the passband version.
        for _, channel in enumerate(processed_two_channels):
            if DEBUG:
                import matplotlib.pyplot as plt

                plt.title(
                    f"Filter type:{filter_type} Roll-off:{rolloff}db/octave Zero-phase:{zero_phase}"
                )
                plt.plot(processed_samples)
                plt.plot(channel, "r--")

                plt.legend(["1D", "2D"])
                plt.show()
            assert np.allclose(channel, processed_samples)


class TestTwoSidedFilterTransforms:
    @pytest.mark.parametrize("filter_type", ["bandstop", "bandpass"])
    @pytest.mark.parametrize(
        "rolloff",
        [6, 12, 18, 120],
    )
    @pytest.mark.parametrize("zero_phase", [False, True])
    def test_one_single_input(self, filter_type, rolloff, zero_phase):

        np.random.seed(1)

        sample_rate = 8000

        samples = get_chirp_test(sample_rate, 40)

        # Parameters for computing periodograms
        nfft = 1024
        nperseg = 1024

        # Expected db drop at f_c's
        if zero_phase:
            expected_db_drop = 6
        else:
            expected_db_drop = 3

        if filter_type == "bandstop":
            FilterTransform = BandStopFilter
        elif filter_type == "bandpass":
            FilterTransform = BandPassFilter

        if zero_phase and rolloff % 12 != 0:
            with pytest.raises(AssertionError):
                augment = FilterTransform(
                    min_center_freq=100.0,
                    max_center_freq=1000.0,
                    min_bandwidth=100.0,
                    max_bandwidth=450.0,
                    min_rolloff=rolloff,
                    max_rolloff=rolloff,
                    zero_phase=zero_phase,
                    p=1.0,
                )
            return
        else:
            augment = FilterTransform(
                min_center_freq=100.0,
                max_center_freq=1000.0,
                min_bandwidth=100.0,
                max_bandwidth=450.0,
                min_rolloff=rolloff,
                max_rolloff=rolloff,
                zero_phase=zero_phase,
                p=1.0,
            )

        processed_samples = augment(samples=samples, sample_rate=sample_rate)
        for n, fc in enumerate(
            [
                augment.parameters["center_freq"] - augment.parameters["bandwidth"] / 2,
                augment.parameters["center_freq"] + augment.parameters["bandwidth"] / 2,
            ]
        ):

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

                plt.title(
                    f"Filter type:{filter_type} Roll-off:{rolloff}db/octave Zero-phase:{zero_phase}"
                )
                plt.plot(wx, 10 * np.log10(np.abs(samples_pxx)))
                plt.plot(wx, 10 * np.log10(np.abs(processed_samples_pxx)), ":")
                plt.legend(["Input signal", f"Highpassed at f_c={fc:.2f}"])
                plt.axvline(fc, color="red", linestyle=":")
                plt.axhline(
                    samples_db_at_fc - (3 if not zero_phase else 6),
                    color="red",
                    linestyle=":",
                )
                plt.xlabel("Frequency (Hz)")
                plt.ylabel("Magnitude (dB)")
                plt.show()

            assert processed_samples.shape == samples.shape
            assert processed_samples.dtype == np.float32

            # Assert that at fc we are at the 3db (6dB at the zero-phase case) point give or take half a db.
            assert np.isclose(
                samples_db_at_fc - processed_sample_db_at_fc,
                expected_db_drop,
                0.5,
            )

            if filter_type == "bandpass":

                if n == 0:
                    # Lower cutoff frequency
                    assert np.isclose(
                        samples_db_below_fc - processed_sample_db_below_fc,
                        expected_db_drop + rolloff,
                        1,
                    )
                else:
                    # Higher cutoff
                    assert np.isclose(
                        samples_db_above_fc, processed_sample_db_below_fc, 0.5
                    )

            elif filter_type == "bandstop":
                if n == 0:
                    assert np.isclose(
                        samples_db_above_fc, processed_sample_db_below_fc, 0.5
                    )
                else:
                    assert np.isclose(
                        samples_db_below_fc - processed_sample_db_below_fc,
                        expected_db_drop + rolloff,
                        1,
                    )

    @pytest.mark.parametrize("filter_type", ["bandpass", "bandstop"])
    @pytest.mark.parametrize(
        "rolloff",
        [12, 120],
    )
    @pytest.mark.parametrize("zero_phase", [False, True])
    def test_two_channel_input(self, filter_type, rolloff, zero_phase):

        sample_rate = 8000
        samples = get_randn_test(sample_rate, 20)

        # Convert to 2D two channels
        two_channels = np.vstack([samples, samples])

        if filter_type == "bandpass":
            FilterTransform = BandPassFilter
        elif filter_type == "bandstop":
            FilterTransform = BandStopFilter

        augment = FilterTransform(
            min_center_freq=1000,
            max_center_freq=1000,
            min_bandwidth=100,
            max_bandwidth=100,
            min_rolloff=rolloff,
            max_rolloff=rolloff,
            zero_phase=zero_phase,
            p=1.0,
        )

        processed_samples = augment(samples=samples, sample_rate=sample_rate)

        processed_two_channels = augment(samples=two_channels, sample_rate=sample_rate)

        assert processed_two_channels.shape[0] == 2
        assert processed_two_channels.shape == two_channels.shape
        assert processed_two_channels.dtype == np.float32

        # Check that the two channels are equal
        assert np.allclose(processed_two_channels[0], processed_two_channels[1])

        # Check that the processed 2D channel version applies the same effect
        # as the passband version.
        for _, channel in enumerate(processed_two_channels):
            if DEBUG:
                import matplotlib.pyplot as plt

                plt.plot(channel)
                plt.plot(processed_samples)
                plt.show()
            assert np.allclose(channel, processed_samples, rtol=1e-5)
