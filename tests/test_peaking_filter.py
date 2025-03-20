import numpy as np
import pytest
import scipy
import scipy.signal

from audiomentations import PeakingFilter
from tests.utils import get_chirp_test, get_randn_test

DEBUG = False


@pytest.mark.parametrize("center_freq", [4000.0])
@pytest.mark.parametrize("gain_db", [-6.0, +6.0])
@pytest.mark.parametrize("q_factor", [1.0])
def test_one_single_input(center_freq, gain_db, q_factor):
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
@pytest.mark.parametrize("num_channels", [1, 2, 3])
def test_multi_channel_input(center_freq, gain_db, q_factor, num_channels):
    sample_rate = 8000
    samples = get_randn_test(sample_rate, 10)

    # Convert to 2D N channels
    n_channels = np.tile(samples, (num_channels, 1))

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

    processed_n_channels = augment(samples=n_channels, sample_rate=sample_rate)

    assert processed_n_channels.shape[0] == num_channels
    assert processed_n_channels.shape == n_channels.shape
    assert processed_n_channels.dtype == np.float32

    # Check that the processed 2D channel version applies the same effect
    # as the passband version.
    for _, channel in enumerate(processed_n_channels):
        if DEBUG:
            import matplotlib.pyplot as plt

            plt.title(f"Filter type: Peaking at {gain_db} with Q={q_factor}")

            plt.plot(processed_samples)
            plt.plot(channel, "r--")

            plt.legend(["1D", "2D"])
            plt.show()
        assert np.allclose(channel, processed_samples)
