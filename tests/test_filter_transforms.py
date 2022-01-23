import numpy as np
import pytest
import scipy
import scipy.signal

from audiomentations.augmentations.transforms import (
    BandPassFilter,
    BandStopFilter,
    HighPassFilter,
    LowPassFilter,
    PeakingFilter,
    LowShelfFilter,
    HighShelfFilter,
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


class TestPeakingFilterTransforms:
    @pytest.mark.parametrize("center_freq", [4000.0])
    @pytest.mark.parametrize("gain_db", [-6.0, +6.0])
    @pytest.mark.parametrize("q_factor", [1.0])
    def test_one_single_input(self, center_freq, gain_db, q_factor):
        np.random.seed(1)

        sample_rate = 16000

        # Parameters for computing periodograms.
        # When examining lower frequencies we need to have
        # a high nfft number.

        nfft = 1024
        nperseg = 1024

        samples = get_chirp_test(sample_rate, 40)

        augment = PeakingFilter(
            min_center_freq=center_freq,
            max_center_freq=center_freq,
            min_gain_db=gain_db,
            max_gain_db=gain_db,
            min_q=q_factor,
            max_q=q_factor,
            p=1.0,
        )

        processed_samples = augment(samples, sample_rate=sample_rate)

        assert processed_samples.shape == samples.shape
        assert processed_samples.dtype == np.float32

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

        center_freq = augment.parameters["center_freq"]

        if DEBUG:
            import matplotlib.pyplot as plt

            plt.title(f"Filter type: Peaking at {gain_db} with Q={q_factor}")
            plt.plot(wx, 10 * np.log10(np.abs(samples_pxx)))
            plt.plot(wx, 10 * np.log10(np.abs(processed_samples_pxx)), ":")
            plt.legend(
                [
                    "Input signal",
                    f"Peaking with center freq {augment.parameters['center_freq']:.2f}",
                ]
            )
            plt.axvline(augment.parameters["center_freq"], color="red", linestyle=":")

            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude (dB)")
            plt.show()

        frequencies_of_interest = np.array([0.0, center_freq, sample_rate / 2])
        expected_differences = np.array([0.0, -gain_db, 0.0])

        for n, freq in enumerate(frequencies_of_interest):
            input_value_db = 10 * np.log10(
                samples_pxx[int(np.round(nfft / sample_rate * freq))]
            )

            output_value_db = 10 * np.log10(
                processed_samples_pxx[int(np.round(nfft / sample_rate * freq))]
            )

            assert np.isclose(
                input_value_db - output_value_db,
                expected_differences[n],
                atol=1.0,
            )

    @pytest.mark.parametrize("center_freq", [10.0, 2000.0, 3900.0])
    @pytest.mark.parametrize("gain_db", [0.0, -6.0, +6.0])
    @pytest.mark.parametrize("q_factor", [0.1, 1.0, 10.0])
    def test_two_channel_input(self, center_freq, gain_db, q_factor):

        sample_rate = 8000
        samples = get_randn_test(sample_rate, 10)

        # Convert to 2D two channels
        two_channels = np.vstack([samples, samples])

        augment = PeakingFilter(
            min_center_freq=center_freq,
            max_center_freq=center_freq,
            min_gain_db=gain_db,
            max_gain_db=gain_db,
            min_q=q_factor,
            max_q=q_factor,
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

                plt.title(f"Filter type: Peaking at {gain_db} with Q={q_factor}")

                plt.plot(processed_samples)
                plt.plot(channel, "r--")

                plt.legend(["1D", "2D"])
                plt.show()
            assert np.allclose(channel, processed_samples)


class TestLowShelfFilterTransform:
    @pytest.mark.parametrize("center_freq", [2000.0, 3900.0])
    @pytest.mark.parametrize("gain_db", [-6.0, +6.0, 0.0])
    @pytest.mark.parametrize("q_factor", [0.1, 1.0])
    def test_one_single_input(self, center_freq, gain_db, q_factor):
        np.random.seed(1)

        sample_rate = 16000

        # Parameters for computing periodograms.
        # When examining lower frequencies we need to have
        # a high nfft number.

        nfft = 4096
        nperseg = 1024

        samples = get_chirp_test(sample_rate, 40)

        augment = LowShelfFilter(
            min_center_freq=center_freq,
            max_center_freq=center_freq,
            min_gain_db=gain_db,
            max_gain_db=gain_db,
            min_q=q_factor,
            max_q=q_factor,
            p=1.0,
        )

        processed_samples = augment(samples, sample_rate=sample_rate)

        assert processed_samples.shape == samples.shape
        assert processed_samples.dtype == np.float32

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

        if DEBUG:
            import matplotlib.pyplot as plt

            plt.title(f"Filter type: High-shelf at {gain_db} with Q={q_factor}")
            plt.plot(wx, 10 * np.log10(np.abs(samples_pxx)))
            plt.plot(wx, 10 * np.log10(np.abs(processed_samples_pxx)), ":")
            plt.legend(
                [
                    "Input signal",
                    f"Peaking with center freq {augment.parameters['center_freq']:.2f}",
                ]
            )
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude (dB)")
            plt.show()

        frequencies_of_interest = np.array([0.0, center_freq, sample_rate / 2])
        expected_differences = np.array([-gain_db, -gain_db / 2, 0.0])

        for n, freq in enumerate(frequencies_of_interest):
            input_value_db = 10 * np.log10(
                samples_pxx[int(np.round(nfft / sample_rate * freq))]
            )

            output_value_db = 10 * np.log10(
                processed_samples_pxx[int(np.round(nfft / sample_rate * freq))]
            )

            assert np.isclose(
                input_value_db - output_value_db,
                expected_differences[n],
                atol=1.0,
            )

    @pytest.mark.parametrize("center_freq", [10.0, 2000.0, 3900.0])
    @pytest.mark.parametrize("gain_db", [0.0, -6.0, +6.0])
    @pytest.mark.parametrize("q_factor", [0.1, 1.0, 10.0])
    def test_two_channel_input(self, center_freq, gain_db, q_factor):

        sample_rate = 8000
        samples = get_randn_test(sample_rate, 10)

        # Convert to 2D two channels
        two_channels = np.vstack([samples, samples])

        augment = PeakingFilter(
            min_center_freq=center_freq,
            max_center_freq=center_freq,
            min_gain_db=gain_db,
            max_gain_db=gain_db,
            min_q=q_factor,
            max_q=q_factor,
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

                plt.title(f"Filter type: Low-shelf at {gain_db} with Q={q_factor}")

                plt.plot(processed_samples)
                plt.plot(channel, "r--")

                plt.legend(["1D", "2D"])
                plt.show()
            assert np.allclose(channel, processed_samples)


class TestHighShelfFilterTransform:
    @pytest.mark.parametrize("center_freq", [2000.0, 3900.0])
    @pytest.mark.parametrize("gain_db", [0.0, -6.0, +6.0])
    @pytest.mark.parametrize("q_factor", [0.1, 1.0])
    def test_one_single_input(self, center_freq, gain_db, q_factor):
        np.random.seed(1)

        sample_rate = 16000

        # Parameters for computing periodograms.
        # When examining lower frequencies we need to have
        # a high nfft number.

        nfft = 4096
        nperseg = 1024

        samples = get_chirp_test(sample_rate, 40)

        augment = HighShelfFilter(
            min_center_freq=center_freq,
            max_center_freq=center_freq,
            min_gain_db=gain_db,
            max_gain_db=gain_db,
            min_q=q_factor,
            max_q=q_factor,
            p=1.0,
        )

        processed_samples = augment(samples, sample_rate=sample_rate)

        assert processed_samples.shape == samples.shape
        assert processed_samples.dtype == np.float32

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

        # Compute gains (db) at the center, dc, and nyquist frequencies
        samples_at_0db_gain = np.max(10 * np.log10(samples_pxx))

        samples_db_at_center_freq = 10 * np.log10(
            samples_pxx[
                int(np.round(nfft / sample_rate * augment.parameters["center_freq"]))
            ]
        )

        # Theoretically, the frequency close to dc should be gain_db, unfortunately
        # we will have abrupt drop at the periodograms at these points so we pick
        # a frequency very close.

        frequency_close_to_dc = 10

        samples_db_at_dc = 10 * np.log10(
            samples_pxx[int(np.round(nfft / sample_rate * frequency_close_to_dc))]
        )

        processed_samples_db_at_dc = 10 * np.log10(
            processed_samples_pxx[
                int(np.round(nfft / sample_rate * frequency_close_to_dc))
            ]
        )

        processed_samples_db_at_center_freq = 10 * np.log10(
            processed_samples_pxx[
                int(np.round(nfft / sample_rate * augment.parameters["center_freq"]))
            ]
        )

        samples_db_at_nyquist = 10 * np.log10(
            samples_pxx[int(np.round(nfft / sample_rate * sample_rate // 2))]
        )

        processed_samples_db_at_nyquist = 10 * np.log10(
            processed_samples_pxx[int(np.round(nfft / sample_rate * sample_rate // 2))]
        )

        # At dc frequency, we should be at 0db gain
        assert np.isclose(samples_db_at_dc, processed_samples_db_at_dc, atol=0.5)

        # At center freq, the output is at half gain
        assert np.isclose(
            samples_db_at_center_freq + gain_db / 2,
            processed_samples_db_at_center_freq,
            atol=0.5,
        )

        # At nyquist, we should be at gain_db
        assert np.isclose(
            samples_db_at_nyquist + gain_db, processed_samples_db_at_nyquist, atol=0.5
        )

        if DEBUG:
            import matplotlib.pyplot as plt

            plt.title(f"Filter type: High-shelf at {gain_db} with Q={q_factor}")
            plt.plot(wx, 10 * np.log10(np.abs(samples_pxx)))
            plt.plot(wx, 10 * np.log10(np.abs(processed_samples_pxx)), ":")
            plt.legend(
                [
                    "Input signal",
                    f"Peaking with center freq {augment.parameters['center_freq']:.2f}",
                ]
            )

            plt.axhline(samples_at_0db_gain + gain_db, color="red", linestyle=":")

            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude (dB)")
            plt.show()

    @pytest.mark.parametrize("center_freq", [10.0, 2000.0, 3900.0])
    @pytest.mark.parametrize("gain_db", [0.0, -6.0, +6.0])
    @pytest.mark.parametrize("q_factor", [0.1, 1.0, 10.0])
    def test_two_channel_input(self, center_freq, gain_db, q_factor):

        sample_rate = 8000
        samples = get_randn_test(sample_rate, 10)

        # Convert to 2D two channels
        two_channels = np.vstack([samples, samples])

        augment = PeakingFilter(
            min_center_freq=center_freq,
            max_center_freq=center_freq,
            min_gain_db=gain_db,
            max_gain_db=gain_db,
            min_q=q_factor,
            max_q=q_factor,
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

                plt.title(f"Filter type: Low-shelf at {gain_db} with Q={q_factor}")

                plt.plot(processed_samples)
                plt.plot(channel, "r--")

                plt.legend(["1D", "2D"])
                plt.show()
            assert np.allclose(channel, processed_samples)


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

        # Parameters for computing periodograms.
        # When examining lower frequencies we need to have
        # a high nfft number.

        nfft = 4096
        nperseg = 1024

        samples = get_chirp_test(sample_rate, 40)

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
                    min_cutoff_freq=1000,
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
            atol=0.5,
        )

        if filter_type == "highpass":
            # Assert the point at half fc (stopband) is around 3 (6dB at the zero-phase case) + <rolloff> dB
            assert np.isclose(
                samples_db_below_fc - processed_sample_db_below_fc,
                expected_db_drop + rolloff,
                1,
            )

            # Assert the point at double fc (passband) is at greater db than at fc
            assert processed_sample_db_at_fc < processed_sample_db_above_fc

        elif filter_type == "lowpass":
            # Assert the point at half fc (stopband) is around 3 (or 6) + <rolloff> dB
            assert np.isclose(
                samples_db_above_fc - processed_sample_db_above_fc,
                expected_db_drop + rolloff,
                1,
            )

            # Assert the point at double fc (passband) at greater db than at fc
            assert processed_sample_db_at_fc < processed_sample_db_below_fc

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
    @pytest.mark.parametrize("center_freq", [500, 1000])
    @pytest.mark.parametrize("bandwidth", [200])
    @pytest.mark.parametrize(
        "rolloff",
        [12, 24],
    )
    @pytest.mark.parametrize("zero_phase", [False, True])
    def test_one_single_input(
        self, filter_type, center_freq, bandwidth, rolloff, zero_phase
    ):

        np.random.seed(1)

        sample_rate = 8000

        # Parameters for computing periodograms.
        # When examining lower frequencies we need to have
        # a high nfft number.

        nfft = 4096
        nperseg = 1024

        samples = get_chirp_test(sample_rate, 40)

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
                    min_center_freq=center_freq,
                    max_center_freq=center_freq,
                    min_bandwidth=bandwidth,
                    max_bandwidth=bandwidth,
                    min_rolloff=rolloff,
                    max_rolloff=rolloff,
                    zero_phase=zero_phase,
                    p=1.0,
                )
            return
        else:
            augment = FilterTransform(
                min_center_freq=center_freq,
                max_center_freq=center_freq,
                min_bandwidth=bandwidth,
                max_bandwidth=bandwidth,
                min_rolloff=rolloff,
                max_rolloff=rolloff,
                zero_phase=zero_phase,
                p=1.0,
            )

        processed_samples = augment(samples=samples, sample_rate=sample_rate)

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
        samples_db_at_center_freq = 10 * np.log10(
            samples_pxx[int(np.round(nfft / sample_rate * center_freq))]
        )
        processed_samples_db_at_center_freq = 10 * np.log10(
            processed_samples_pxx[int(np.round(nfft / sample_rate * center_freq))]
        )

        if DEBUG:
            import matplotlib.pyplot as plt

            plt.title(
                f"Filter type:{filter_type} Roll-off:{rolloff}db/octave Zero-phase:{zero_phase}"
            )
            plt.plot(wx, 10 * np.log10(np.abs(samples_pxx)))
            plt.plot(wx, 10 * np.log10(np.abs(processed_samples_pxx)), ":")
            plt.axvline(center_freq, color="red", linestyle=":")

            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude (dB)")
            plt.show()

        for n, fc in enumerate(
            [
                augment.parameters["center_freq"] - augment.parameters["bandwidth"] / 2,
                augment.parameters["center_freq"] + augment.parameters["bandwidth"] / 2,
            ]
        ):

            # Compute db at cutoffs at the input as well as the filtered signals
            samples_db_at_fc = 10 * np.log10(
                samples_pxx[int(np.round(nfft / sample_rate * fc))]
            )

            samples_db_below_fc = 10 * np.log10(
                samples_pxx[int(np.round(nfft / sample_rate * fc / 2))]
            )

            processed_sample_db_at_fc = 10 * np.log10(
                processed_samples_pxx[int(np.round(nfft / sample_rate * fc))]
            )

            processed_sample_db_below_fc = 10 * np.log10(
                processed_samples_pxx[int(np.round(nfft / sample_rate * fc / 2))]
            )

            assert processed_samples.shape == samples.shape
            assert processed_samples.dtype == np.float32

            # Assert that at fc we are at the 3db (6dB at the zero-phase case) point give or take half a db.
            assert np.isclose(
                samples_db_at_fc - processed_sample_db_at_fc,
                expected_db_drop,
                atol=1,
            )

            # For all cases below, f_c is the cutoff point, Compare region A for
            # the original and the processed version.
            if filter_type == "bandpass":

                if n == 0:
                    #      -------
                    # ..../       \.....
                    #    ^
                    # A  |
                    #   f_c

                    assert samples_db_below_fc > processed_sample_db_below_fc
                else:
                    #      -------
                    # ..../       \.....
                    #             ^
                    #          A  |
                    #            f_c

                    assert np.isclose(
                        samples_db_at_center_freq,
                        processed_samples_db_at_center_freq,
                        atol=1,
                    )

            elif filter_type == "bandstop":
                if n == 0:
                    # ---           ----
                    #    \_________/
                    #    ^
                    # A  |
                    #   f_c
                    assert np.isclose(
                        samples_db_below_fc,
                        processed_sample_db_below_fc,
                        atol=1.0,
                    )

                else:
                    # ---           ----
                    #    \_________/
                    #              ^
                    #           A  |
                    #             f_c
                    assert (
                        samples_db_at_center_freq > processed_samples_db_at_center_freq
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
