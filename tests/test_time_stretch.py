import numpy as np
import pytest

from audiomentations import TimeStretch


@pytest.mark.parametrize("method", ["librosa_phase_vocoder", "signalsmith_stretch"])
def test_dynamic_length(method):
    samples = np.zeros((2048,), dtype=np.float32)
    sample_rate = 16000
    augmenter = TimeStretch(
        min_rate=0.8, max_rate=0.9, leave_length_unchanged=False, method=method, p=1.0
    )

    samples_out = augmenter(samples=samples, sample_rate=sample_rate)

    assert samples_out.dtype == np.float32
    assert samples_out.ndim == samples.ndim
    assert samples_out.shape[-1] > 2048


@pytest.mark.parametrize("method", ["librosa_phase_vocoder", "signalsmith_stretch"])
def test_fixed_length(method):
    samples = np.zeros((2048,), dtype=np.float32)
    sample_rate = 16000
    augmenter = TimeStretch(
        min_rate=0.8, max_rate=0.9, leave_length_unchanged=True, method=method, p=1.0
    )

    samples_out = augmenter(samples=samples, sample_rate=sample_rate)

    assert samples_out.dtype == np.float32
    assert samples_out.ndim == samples.ndim
    assert samples_out.shape[-1] == 2048


@pytest.mark.parametrize("method", ["librosa_phase_vocoder", "signalsmith_stretch"])
def test_multichannel(method):
    num_channels = 3
    samples = np.random.normal(0, 0.1, size=(num_channels, 5555)).astype(np.float32)
    sample_rate = 16000
    augmenter = TimeStretch(
        min_rate=0.8, max_rate=0.9, leave_length_unchanged=True, method=method, p=1.0
    )

    samples_out = augmenter(samples=samples, sample_rate=sample_rate)

    assert samples.dtype == samples_out.dtype
    assert samples.shape == samples_out.shape
    for i in range(num_channels):
        assert not np.allclose(samples[i], samples_out[i])


def test_invalid_params():
    with pytest.raises(ValueError):
        TimeStretch(min_rate=0.0)
    with pytest.raises(ValueError):
        TimeStretch(max_rate=11.0)
    with pytest.raises(ValueError):
        TimeStretch(min_rate=2.0, max_rate=1.0)
    with pytest.raises(ValueError):
        TimeStretch(min_rate=1.0, max_rate=2.0, method="invalid")
