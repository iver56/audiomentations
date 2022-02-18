import numpy as np
import pytest

from audiomentations import Mp3Compression, Compose


class TestMp3Compression:
    @pytest.mark.parametrize(
        "backend",
        ["pydub", "lameenc"],
    )
    @pytest.mark.parametrize(
        "shape",
        [(44100,), (1, 22049), (2, 10000)],
    )
    def test_apply_mp3_compression(self, backend: str, shape: tuple):
        samples_in = np.random.normal(0, 1, size=shape).astype(np.float32)
        sample_rate = 44100
        augmenter = Compose(
            [Mp3Compression(p=1.0, min_bitrate=48, max_bitrate=48, backend=backend)]
        )

        samples_out = augmenter(samples=samples_in, sample_rate=sample_rate)
        assert len(shape) == len(samples_out.shape)
        assert samples_out.dtype == np.float32
        assert samples_out.shape[-1] >= shape[-1]
        assert samples_out.shape[-1] < shape[-1] + 3000
        if len(shape) == 2:
            assert samples_out.shape[0] == shape[0]

    @pytest.mark.parametrize(
        "backend",
        ["pydub", "lameenc"],
    )
    @pytest.mark.parametrize(
        "shape",
        [(16000,), (1, 12049), (2, 5000)],
    )
    def test_apply_mp3_compression_low_bitrate(self, backend: str, shape: tuple):
        samples_in = np.random.normal(0, 1, size=shape).astype(np.float32)
        sample_rate = 16000
        augmenter = Compose(
            [Mp3Compression(p=1.0, min_bitrate=8, max_bitrate=8, backend=backend)]
        )

        samples_out = augmenter(samples=samples_in, sample_rate=sample_rate)
        assert len(shape) == len(samples_out.shape)
        assert samples_out.dtype == np.float32
        assert samples_out.shape[-1] >= shape[-1]
        assert samples_out.shape[-1] < shape[-1] + 3100
        if len(shape) == 2:
            assert samples_out.shape[0] == shape[0]

    def test_invalid_argument_combination(self):
        with pytest.raises(AssertionError):
            _ = Mp3Compression(min_bitrate=400, max_bitrate=800)

        with pytest.raises(AssertionError):
            _ = Mp3Compression(min_bitrate=2, max_bitrate=4)

        with pytest.raises(AssertionError):
            _ = Mp3Compression(min_bitrate=64, max_bitrate=8)
