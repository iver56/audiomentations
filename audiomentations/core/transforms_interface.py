import random
import warnings

import numpy as np

from audiomentations.core.utils import (
    is_waveform_multichannel,
    is_spectrogram_multichannel,
)


class MultichannelAudioNotSupportedException(Exception):
    pass


class MonoAudioNotSupportedException(Exception):
    pass


class BaseTransform:
    supports_mono = True
    supports_multichannel = False

    def __init__(self, p=0.5):
        assert 0 <= p <= 1
        self.p = p
        self.parameters = {"should_apply": None}
        self.are_parameters_frozen = False

    def serialize_parameters(self):
        """Return the parameters as a JSON-serializable dict."""
        return self.parameters

    def freeze_parameters(self):
        """
        Mark all parameters as frozen, i.e. do not randomize them for each call. This can be
        useful if you want to apply an effect with the exact same parameters to multiple sounds.
        """
        self.are_parameters_frozen = True

    def unfreeze_parameters(self):
        """
        Unmark all parameters as frozen, i.e. let them be randomized for each call.
        """
        self.are_parameters_frozen = False


class BaseWaveformTransform(BaseTransform):
    def apply(self, samples, sample_rate):
        raise NotImplementedError

    def is_multichannel(self, samples):
        return is_waveform_multichannel(samples)

    def __call__(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        if samples.dtype == np.float64:
            warnings.warn(
                "Warning: input samples dtype is np.float64. Converting to np.float32"
            )
            samples = np.float32(samples)
        if not self.are_parameters_frozen:
            self.randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"] and len(samples) > 0:
            if self.is_multichannel(samples):
                if samples.shape[0] > samples.shape[1]:
                    warnings.warn(
                        "Multichannel audio must have channels first, not channels last. In"
                        " other words, the shape must be (channels, samples), not"
                        " (samples, channels)"
                    )
                if not self.supports_multichannel:
                    raise MultichannelAudioNotSupportedException(
                        "{} only supports mono audio, not multichannel audio. In other words, a 1-dimensional input"
                        " ndarray was expected, but the input had more than 1 dimension.".format(
                            self.__class__.__name__
                        )
                    )
            elif not self.supports_mono:
                raise MonoAudioNotSupportedException(
                    "{} only supports multichannel audio, not mono audio".format(
                        self.__class__.__name__
                    )
                )
            return self.apply(samples, sample_rate)
        return samples

    def randomize_parameters(self, samples, sample_rate):
        self.parameters["should_apply"] = random.random() < self.p


class BaseSpectrogramTransform(BaseTransform):
    def apply(self, magnitude_spectrogram):
        raise NotImplementedError

    def is_multichannel(self, samples):
        return is_spectrogram_multichannel(samples)

    def __call__(self, magnitude_spectrogram):
        if not self.are_parameters_frozen:
            self.randomize_parameters(magnitude_spectrogram)
        if (
            self.parameters["should_apply"]
            and magnitude_spectrogram.shape[0] > 0
            and magnitude_spectrogram.shape[1] > 0
        ):
            if self.is_multichannel(magnitude_spectrogram):
                """
                if magnitude_spectrogram.shape[0] > magnitude_spectrogram.shape[1]:
                    warnings.warn(
                        "Multichannel audio must have channels first, not channels last"
                    )
                """
                if not self.supports_multichannel:
                    raise MultichannelAudioNotSupportedException(
                        "{} only supports mono audio, not multichannel audio".format(
                            self.__class__.__name__
                        )
                    )
            elif not self.supports_mono:
                raise MonoAudioNotSupportedException(
                    "{} only supports multichannel audio, not mono audio".format(
                        self.__class__.__name__
                    )
                )

            return self.apply(magnitude_spectrogram)
        return magnitude_spectrogram

    def randomize_parameters(self, magnitude_spectrogram):
        self.parameters["should_apply"] = random.random() < self.p
