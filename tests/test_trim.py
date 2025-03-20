import numpy as np

from audiomentations import Trim


def test_trim():
    sample_len = 1024
    samples1 = np.zeros((sample_len,), dtype=np.float32)
    samples2 = np.random.normal(0, 1, size=sample_len).astype(np.float32)
    sample_rate = 16000
    augmenter = Trim(top_db=20, p=1.0)
    samples_in = np.hstack((samples1, samples2))
    assert len(samples_in) == sample_len * 2
    samples_out = augmenter(samples=samples_in, sample_rate=sample_rate)

    assert samples_out.dtype == np.float32
    assert len(samples_out) < sample_len * 2
