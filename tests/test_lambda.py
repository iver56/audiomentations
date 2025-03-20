import numpy as np
import pytest

from audiomentations import Lambda, Gain


def test_gain_lambda():
    samples_in = np.random.normal(0, 1, size=1024).astype(np.float32)
    augmenter = Lambda(
        transform=Gain(min_gain_db=50, max_gain_db=50, p=1.0), p=1.0
    )
    std_in = np.mean(np.abs(samples_in))
    samples_out = augmenter(samples=samples_in, sample_rate=16000)
    std_out = np.mean(np.abs(samples_out))
    assert samples_out.dtype == np.float32
    assert samples_out.shape == samples_in.shape
    assert std_out > 100 * std_in

def test_lambda_with_kwargs():
    samples_in = np.random.normal(0, 1, size=1024).astype(np.float32)
    augmenter = Lambda(
        transform=lambda samples, sample_rate, offset: samples + offset,
        p=1.0,
        offset=-0.2,
    )
    input_mean = np.mean(samples_in)
    samples_out = augmenter(samples=samples_in, sample_rate=16000)
    output_mean = np.mean(samples_out)
    assert samples_out.dtype == np.float32
    assert samples_out.shape == samples_in.shape
    assert output_mean == pytest.approx(input_mean - 0.2)
