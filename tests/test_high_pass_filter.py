import numpy as np
import pytest
import scipy
import scipy.signal

from audiomentations import HighPassFilter
from tests.utils import get_chirp_test, get_randn_test

DEBUG = False


@pytest.mark.parametrize("cutoff_frequency", [1000])
@pytest.mark.parametrize("rolloff", [6, 24])
@pytest.mark.parametrize("zero_phase", [False, True])
def test_one_single_input(cutoff_frequency, rolloff, zero_phase):
    sample_rate = 8000

    # Parameters for computing periodograms.
    # When examining lower frequencies we need to have
    # a high nfft number.

    nfft = 2048 * 2
    nperseg = 128

    samples = get_chirp_test(sample_rate, 10)

    # Expected dB drop at fc
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
            f"Filter type: High Roll-off:{rolloff}dB/octave Zero-phase:{zero_phase}"
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
    expected_differences = np.array([expected_db_drop + rolloff, expected_db_drop, 0.0])

    # Tolerances for the differences in dB
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


@pytest.mark.parametrize("samples", [get_chirp_test(8000, 40)])
@pytest.mark.parametrize("rolloff", [12, 120])
@pytest.mark.parametrize("zero_phase", [False, True])
@pytest.mark.parametrize("num_channels", [1, 2, 3])
def test_n_channel_input(samples, rolloff, zero_phase, num_channels):
    sample_rate = 8000
    samples = get_randn_test(sample_rate, 10)

    # Convert to 2D N channels
    n_channels = np.tile(samples, (num_channels, 1))

    augment = HighPassFilter(
        min_cutoff_freq=1000,
        max_cutoff_freq=1000,
        min_rolloff=rolloff,
        max_rolloff=rolloff,
        zero_phase=zero_phase,
        p=1.0,
    )

    processed_samples = augment(samples=samples, sample_rate=sample_rate)

    processed_n_channels = augment(samples=n_channels, sample_rate=sample_rate)

    assert processed_n_channels.shape[0] == num_channels
    assert processed_n_channels.shape == n_channels.shape
    assert processed_n_channels.dtype == np.float32

    # Check that the processed 2D channel version applies the same effect
    # as the passband version.
    for _, channel in enumerate(processed_n_channels):
        if DEBUG:
            import matplotlib.pyplot as plt

            plt.title(
                f"Filter type: Highpass Roll-off:{rolloff}dB/octave Zero-phase:{zero_phase}"
            )
            plt.plot(processed_samples)
            plt.plot(channel, "r--")

            plt.legend(["1D", "2D"])
            plt.show()
        assert np.allclose(channel, processed_samples)


@pytest.mark.parametrize("cutoff_frequency", [5000])
@pytest.mark.parametrize("rolloff", [6])
@pytest.mark.parametrize("zero_phase", [False])
def test_nyquist_limit(cutoff_frequency, rolloff, zero_phase):
    # Test that the filter doesn't raise an exception when
    # cutoff_frequency is greater than the Nyquist frequency

    sample_rate = 8000

    samples = get_chirp_test(sample_rate, 3)

    augment = HighPassFilter(
        min_cutoff_freq=cutoff_frequency,
        max_cutoff_freq=cutoff_frequency,
        min_rolloff=rolloff,
        max_rolloff=rolloff,
        zero_phase=zero_phase,
        p=1.0,
    )

    processed_samples = augment(samples=samples, sample_rate=sample_rate)
    assert processed_samples.shape == samples.shape


def test_validation():
    with pytest.raises(ValueError):
        HighPassFilter(min_cutoff_freq=0)
