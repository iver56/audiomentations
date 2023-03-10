import numpy as np

from audiomentations.core.transforms_interface import BaseWaveformTransform


class Normalize(BaseWaveformTransform):
    """
    Apply a constant amount of gain, so that highest signal level present in the sound becomes
    0 dBFS, i.e. the loudest level allowed if all samples must be between -1 and 1. Also known
    as peak normalization.
    """

    supports_multichannel = True

    def __init__(self, apply_to: str = "all", p: float = 0.5):
        super().__init__(p)
        assert apply_to in ("all", "only_too_loud_sounds")
        self.apply_to = apply_to

    def randomize_parameters(self, samples: np.ndarray, sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["max_amplitude"] = np.amax(np.abs(samples))

    def apply(self, samples: np.ndarray, sample_rate: int):
        if (
            self.apply_to == "only_too_loud_sounds"
            and self.parameters["max_amplitude"] < 1.0
        ):
            return samples

        if self.parameters["max_amplitude"] > 0:
            return samples / self.parameters["max_amplitude"]
        else:
            return samples
