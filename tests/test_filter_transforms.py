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
        expected_differences = np.array([0, -gain_db / 2, -gain_db])

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


class TestHighPassFilterTransform:
    @pytest.mark.parametrize("cutoff_frequency", [1000])
    @pytest.mark.parametrize(
        "rolloff",
        [6, 24],
    )
    @pytest.mark.parametrize("zero_phase", [False, True])
    def test_one_single_input(self, cutoff_frequency, rolloff, zero_phase):

        sample_rate = 8000

        # Parameters for computing periodograms.
        # When examining lower frequencies we need to have
        # a high nfft number.

        nfft = 2048 * 2
        nperseg = 128

        samples = get_chirp_test(sample_rate, 10)

        # Expected db drop at fc
        if zero_phase:
            expected_db_drop = 6
        else:
            expected_db_drop = 3

        if zero_phase and rolloff % 12 != 0:
            with pytest.raises(AssertionError):
                augment = HighPassFilter(
                    min_cutoff_freq=cutoff_frequency,
                    max_cutoff_freq=cutoff_frequency,
                    min_rolloff=rolloff,
                    max_rolloff=rolloff,
                    zero_phase=zero_phase,
                    p=1.0,
                )
            return
        else:
            augment = HighPassFilter(
                min_cutoff_freq=cutoff_frequency,
                max_cutoff_freq=cutoff_frequency,
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

        if DEBUG:
            import matplotlib.pyplot as plt

            plt.title(
                f"Filter type: High Roll-off:{rolloff}db/octave Zero-phase:{zero_phase}"
            )
            plt.plot(wx, 10 * np.log10(np.abs(samples_pxx)))
            plt.plot(wx, 10 * np.log10(np.abs(processed_samples_pxx)), ":")
            plt.legend(["Input signal", f"Highpassed at f_c={cutoff_frequency:.2f}"])
            plt.axvline(cutoff_frequency, color="red", linestyle=":")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude (dB)")
            plt.show()

        assert processed_samples.shape == samples.shape
        assert processed_samples.dtype == np.float32

        frequencies_of_interest = np.array(
            [cutoff_frequency / 2, cutoff_frequency, sample_rate / 2]
        )
        expected_differences = np.array(
            [expected_db_drop + rolloff, expected_db_drop, 0.0]
        )

        # Tolerances for the differences in db
        tolerances = np.array([5.0, 2.0, 1.0])

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
                atol=tolerances[n],
            )

    @pytest.mark.parametrize(
        "samples",
        [get_chirp_test(8000, 40)],
    )
    @pytest.mark.parametrize(
        "rolloff",
        [12, 120],
    )
    @pytest.mark.parametrize("zero_phase", [False, True])
    def test_two_channel_input(self, samples, rolloff, zero_phase):

        sample_rate = 8000
        samples = get_randn_test(sample_rate, 10)

        # Convert to 2D two channels
        two_channels = np.vstack([samples, samples])

        augment = HighPassFilter(
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
                    f"Filter type: Highpass Roll-off:{rolloff}db/octave Zero-phase:{zero_phase}"
                )
                plt.plot(processed_samples)
                plt.plot(channel, "r--")

                plt.legend(["1D", "2D"])
                plt.show()
            assert np.allclose(channel, processed_samples)


class TestBandPassFilterTransform:
    @pytest.mark.parametrize("center_frequency", [3000])
    @pytest.mark.parametrize("bandwidth", [2000])
    @pytest.mark.parametrize(
        "rolloff",
        [6, 24],
    )
    @pytest.mark.parametrize("zero_phase", [False, True])
    def test_one_single_input(self, center_frequency, bandwidth, rolloff, zero_phase):

        sample_rate = 16000

        # Parameters for computing periodograms.
        # When examining lower frequencies we need to have
        # a high nfft number.

        nfft = 2048 * 2
        nperseg = 128

        samples = get_chirp_test(sample_rate, 10)

        # Expected db drop at fc
        if zero_phase:
            expected_db_drop = 6
        else:
            expected_db_drop = 3

        if zero_phase and rolloff % 12 != 0:
            with pytest.raises(AssertionError):
                augment = BandPassFilter(
                    min_center_freq=center_frequency,
                    max_center_freq=center_frequency,
                    min_bandwidth=bandwidth,
                    max_bandwidth=bandwidth,
                    min_rolloff=rolloff,
                    max_rolloff=rolloff,
                    zero_phase=zero_phase,
                    p=1.0,
                )
            return
        else:
            augment = BandPassFilter(
                min_center_freq=center_frequency,
                max_center_freq=center_frequency,
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

        if DEBUG:
            import matplotlib.pyplot as plt

            plt.title(
                f"Filter type: Band-pass Roll-off:{rolloff}db/octave Zero-phase:{zero_phase}"
            )
            plt.plot(wx, 10 * np.log10(np.abs(samples_pxx)))
            plt.plot(wx, 10 * np.log10(np.abs(processed_samples_pxx)), ":")
            plt.legend(["Input signal", f"Highpassed at f_c={center_frequency:.2f}"])
            plt.axvline(center_frequency, color="red", linestyle=":")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude (dB)")
            plt.show()

        assert processed_samples.shape == samples.shape
        assert processed_samples.dtype == np.float32

        left_cutoff_freq = center_frequency - bandwidth / 2
        right_cutoff_freq = center_frequency + bandwidth / 2

        frequencies_of_interest = np.array(
            [
                left_cutoff_freq,
                center_frequency,
                right_cutoff_freq,
            ]
        )
        expected_differences = np.array(
            [
                expected_db_drop,
                0,
                expected_db_drop,
            ]
        )

        # Tolerances for the differences in db
        tolerances = np.array([2.0, 1.0, 2.0])

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
                atol=tolerances[n],
            )

    @pytest.mark.parametrize("center_frequency", [3000])
    @pytest.mark.parametrize("bandwidth", [2000])
    @pytest.mark.parametrize(
        "samples",
        [get_chirp_test(8000, 40)],
    )
    @pytest.mark.parametrize(
        "rolloff",
        [12, 120],
    )
    @pytest.mark.parametrize("zero_phase", [False, True])
    def test_two_channel_input(
        self, center_frequency, bandwidth, samples, rolloff, zero_phase
    ):

        sample_rate = 16000
        samples = get_randn_test(sample_rate, 10)

        # Convert to 2D two channels
        two_channels = np.vstack([samples, samples])

        augment = BandPassFilter(
            min_center_freq=center_frequency,
            max_center_freq=center_frequency,
            min_bandwidth=bandwidth,
            max_bandwidth=bandwidth,
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
                    f"Filter type: Band-pass Roll-off:{rolloff}db/octave Zero-phase:{zero_phase}"
                )
                plt.plot(processed_samples)
                plt.plot(channel, "r--")

                plt.legend(["1D", "2D"])
                plt.show()
            assert np.allclose(channel, processed_samples)


class TestBandStopFilterTransform:
    @pytest.mark.parametrize("center_frequency", [3000])
    @pytest.mark.parametrize("bandwidth", [2000])
    @pytest.mark.parametrize(
        "rolloff",
        [6, 24],
    )
    @pytest.mark.parametrize("zero_phase", [False, True])
    def test_one_single_input(self, center_frequency, bandwidth, rolloff, zero_phase):

        sample_rate = 16000

        # Parameters for computing periodograms.
        # When examining lower frequencies we need to have
        # a high nfft number.

        nfft = 2048 * 2
        nperseg = 128

        samples = get_chirp_test(sample_rate, 10)

        # Expected db drop at fc
        if zero_phase:
            expected_db_drop = 6
        else:
            expected_db_drop = 3

        if zero_phase and rolloff % 12 != 0:
            with pytest.raises(AssertionError):
                augment = BandStopFilter(
                    min_center_freq=center_frequency,
                    max_center_freq=center_frequency,
                    min_bandwidth=bandwidth,
                    max_bandwidth=bandwidth,
                    min_rolloff=rolloff,
                    max_rolloff=rolloff,
                    zero_phase=zero_phase,
                    p=1.0,
                )
            return
        else:
            augment = BandStopFilter(
                min_center_freq=center_frequency,
                max_center_freq=center_frequency,
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

        if DEBUG:
            import matplotlib.pyplot as plt

            plt.title(
                f"Filter type: Band-stop Roll-off:{rolloff}db/octave Zero-phase:{zero_phase}"
            )
            plt.plot(wx, 10 * np.log10(np.abs(samples_pxx)))
            plt.plot(wx, 10 * np.log10(np.abs(processed_samples_pxx)), ":")
            plt.legend(["Input signal", f"Highpassed at f_c={center_frequency:.2f}"])
            plt.axvline(center_frequency, color="red", linestyle=":")
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Magnitude (dB)")
            plt.show()

        assert processed_samples.shape == samples.shape
        assert processed_samples.dtype == np.float32

        left_cutoff_freq = center_frequency - bandwidth / 2
        right_cutoff_freq = center_frequency + bandwidth / 2

        frequencies_of_interest = np.array(
            [0, left_cutoff_freq, right_cutoff_freq, sample_rate / 2]
        )
        expected_differences = np.array([0, expected_db_drop, expected_db_drop, 0.0])

        # Tolerances for the differences in db
        tolerances = np.array([1.0, 2.0, 2.0, 1.0])

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
                atol=tolerances[n],
            )

    @pytest.mark.parametrize("center_frequency", [3000])
    @pytest.mark.parametrize("bandwidth", [2000])
    @pytest.mark.parametrize(
        "samples",
        [get_chirp_test(8000, 40)],
    )
    @pytest.mark.parametrize(
        "rolloff",
        [12, 120],
    )
    @pytest.mark.parametrize("zero_phase", [False, True])
    def test_two_channel_input(
        self, center_frequency, bandwidth, samples, rolloff, zero_phase
    ):

        sample_rate = 16000
        samples = get_randn_test(sample_rate, 10)

        # Convert to 2D two channels
        two_channels = np.vstack([samples, samples])

        augment = BandStopFilter(
            min_center_freq=center_frequency,
            max_center_freq=center_frequency,
            min_bandwidth=bandwidth,
            max_bandwidth=bandwidth,
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
                    f"Filter type: Band-stop Roll-off:{rolloff}db/octave Zero-phase:{zero_phase}"
                )
                plt.plot(processed_samples)
                plt.plot(channel, "r--")

                plt.legend(["1D", "2D"])
                plt.show()
            assert np.allclose(channel, processed_samples)
