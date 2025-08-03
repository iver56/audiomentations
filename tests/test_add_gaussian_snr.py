import numpy as np
import pytest

from audiomentations import AddGaussianSNR


def test_gaussian_noise_snr_defaults():
    np.random.seed(42)
    samples_in = np.random.normal(0, 1, size=1024).astype(np.float32)
    augmenter = AddGaussianSNR(p=1.0)
    std_in = np.mean(np.abs(samples_in))
    samples_out = augmenter(samples=samples_in, sample_rate=16000)
    std_out = np.mean(np.abs(samples_out))
    assert samples_out.dtype == np.float32
    assert not (float(std_out) == pytest.approx(0.0))
    assert std_out > std_in


def test_gaussian_noise_snr():
    np.random.seed(42)
    samples_in = np.random.normal(0, 1, size=1024).astype(np.float32)
    augmenter = AddGaussianSNR(min_snr_db=15, max_snr_db=35, p=1.0)
    std_in = np.mean(np.abs(samples_in))
    samples_out = augmenter(samples=samples_in, sample_rate=16000)
    std_out = np.mean(np.abs(samples_out))
    assert samples_out.dtype == np.float32
    assert not (float(std_out) == pytest.approx(0.0))
    assert std_out > std_in


def test_gaussian_noise_snr_multichannel():
    np.random.seed(42)
    samples = np.random.normal(0, 0.1, size=(3, 8888)).astype(np.float32)
    augmenter = AddGaussianSNR(min_snr_db=15, max_snr_db=35, p=1.0)
    samples_out = augmenter(samples=samples, sample_rate=16000)

    assert samples_out.dtype == np.float32
    assert float(np.sum(np.abs(samples_out))) > float(np.sum(np.abs(samples)))


def test_validation():
    with pytest.raises(ValueError):
        AddGaussianSNR(min_snr_db=40.0, max_snr_db=20.0)
