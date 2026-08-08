import numpy as np
from numpy.typing import NDArray

import librosa

from audiomentations.core.transforms_interface import BaseWaveformTransform


class SpectralGating(BaseWaveformTransform):
    """
    Applies a dynamic magnitude threshold to an audio STFT and reconstructs it.
    """

    supports_multichannel = True

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        percentile_range: tuple[float, float] = (5.0, 40.0),
        p: float = 0.5
    ):
        """
        :param n_fft: Length of the FFT window
        :param hop_length: Number of samples between successive frames
        :percentile_range: (min, max) range to pick a random percentile threshold
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        self.n_fft = n_fft
        self.hop_length = hop_length
        assert percentile_range[0] < percentile_range[1]
        self.percentile_range = percentile_range

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["target_percentile"] = np.random.uniform(self.percentile_range[0], self.percentile_range[1])

    def apply(
        self, samples: NDArray[np.float32], sample_rate: int
    ) -> NDArray[np.float32]:
    
        assert self.n_fft <= samples.shape[-1]
        
        stft_matrix = librosa.stft(samples, n_fft=self.n_fft, hop_length=self.hop_length)
        mag, phase = librosa.magphase(stft_matrix)
        
        threshold = float(np.percentile(mag, self.parameters["target_percentile"]))
        mag_filtered = np.where(mag < threshold, 0.0, mag)

        samples = librosa.istft(mag_filtered * phase, hop_length=self.hop_length)
        return samples

