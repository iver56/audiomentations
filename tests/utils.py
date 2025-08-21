from typing import Optional

import numpy as np
import scipy
import scipy.signal
from numpy.typing import NDArray


def plot_matrix(matrix, output_image_path=None, vmin=None, vmax=None, title=None):
    """
    Plot a 2D matrix with viridis color map

    :param matrix: 2D numpy array
    :return:
    """
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    if title is not None:
        ax.set_title(title)
    plt.imshow(matrix, vmin=vmin, vmax=vmax)
    if matrix.shape[-1] != 3:
        plt.colorbar()
    if output_image_path:
        plt.savefig(str(output_image_path), dpi=200)
    else:
        plt.show()
    plt.close(fig)


def plot_waveforms(wf1, wf2=None, wf3=None, title="Untitled"):
    """Plot one, two or three short 1D waveforms. Useful for debugging."""
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)
    plt.plot(wf1, label="wf1", alpha=0.66)
    if wf2 is not None:
        plt.plot(wf2, label="wf2", alpha=0.66)
    if wf3 is not None:
        plt.plot(wf3, label="wf3", alpha=0.66)
    plt.legend()
    plt.show()
    plt.close(fig)


def get_chirp_test(sample_rate, duration):
    """Create a `duration` seconds chirp from 0Hz to `Nyquist frequency`"""
    n = np.arange(0, duration, 1 / sample_rate)
    samples = scipy.signal.chirp(n, 0, duration, sample_rate // 2, method="linear")
    return samples.astype(np.float32)


def get_randn_test(sample_rate, duration):
    """Create a random noise test stimulus"""
    n_samples = int(duration * sample_rate)
    samples = np.random.randn(n_samples)
    return samples.astype(np.float32)


def fast_autocorr(original: NDArray, delayed: NDArray, t: int = 1):
    """Only every 4th sample is considered in order to improve execution time"""
    if t == 0:
        return np.corrcoef([original[::4], delayed[::4]])[1, 0]
    elif t < 0:
        return np.corrcoef([original[-t::4], delayed[:t:4]])[1, 0]
    else:
        return np.corrcoef([original[:-t:4], delayed[t::4]])[1, 0]


def find_best_alignment_offset_with_corr_coef(
    reference_signal: NDArray[np.float32],
    delayed_signal: NDArray[np.float32],
    min_offset_samples: int,
    max_offset_samples: int,
    lookahead_samples: int | None = None,
    consider_both_polarities: bool = True,
):
    """
    Returns the estimated delay (in samples) between the original and delayed signal,
    calculated using correlation coefficients. The delay is optimized to maximize the
    correlation between the signals.

    Args:
        reference_signal (NDArray[np.float32]): The original signal array.
        delayed_signal (NDArray[np.float32]): The delayed signal array.
        min_offset_samples (int): The minimum delay offset to consider, in samples.
                                  Can be negative.
        max_offset_samples (int): The maximum delay offset to consider, in samples.
        lookahead_samples (Optional[int]): The number of samples to look at
                                           while estimating the delay. If None, the
                                           whole delayed signal is considered.
        consider_both_polarities (bool): If True, the function will consider both positive
                                         and negative correlations, which corresponds to
                                         the same or opposite polarities in signals,
                                         respectively. Defaults to True.

    Returns:
        tuple: Estimated delay (int) and correlation coefficient (float).
    """
    if lookahead_samples is not None and len(reference_signal) > lookahead_samples:
        middle_of_signal_index = int(np.floor(len(reference_signal) / 2))
        original_signal_slice = reference_signal[
            middle_of_signal_index : middle_of_signal_index + lookahead_samples
        ]
        delayed_signal_slice = delayed_signal[
            middle_of_signal_index : middle_of_signal_index + lookahead_samples
        ]
    else:
        original_signal_slice = reference_signal
        delayed_signal_slice = delayed_signal

    coefs = []
    for lag in range(min_offset_samples, max_offset_samples):
        correlation_coef = fast_autocorr(
            original_signal_slice, delayed_signal_slice, t=lag
        )
        coefs.append(correlation_coef)

    if consider_both_polarities:
        # In this mode we aim to find the correlation coefficient of highest magnitude.
        # We do this to consider the possibility that the delayed signal has opposite
        # polarity compared to the original signal, in which case the correlation
        # coefficient would be negative.
        most_extreme_coef_index = int(np.argmax(np.abs(coefs)))
    else:
        most_extreme_coef_index = int(np.argmax(coefs))
    most_extreme_coef = coefs[most_extreme_coef_index]
    offset = most_extreme_coef_index + min_offset_samples
    return offset, most_extreme_coef
