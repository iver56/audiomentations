import numpy as np
import scipy
import scipy.signal


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
