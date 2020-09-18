import random
import warnings

from audiomentations.core.utils import is_multichannel


class MultichannelAudioNotSupportedException(Exception):
    pass


class BasicTransform:
    supports_multichannel = False

    def __init__(self, p=0.5):
        assert 0 <= p <= 1
        self.p = p
        self.parameters = {"should_apply": None}
        self.are_parameters_frozen = False

    def __call__(self, samples, sample_rate):
        if not self.are_parameters_frozen:
            self.randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"] and len(samples) > 0:
            if is_multichannel(samples):
                if samples.shape[0] > samples.shape[1]:
                    warnings.warn(
                        "Multichannel audio must have channels first, not channels last. In"
                        " other words, the shape must be (channels, samples), not"
                        " (samples, channels)"
                    )
                if not self.supports_multichannel:
                    raise MultichannelAudioNotSupportedException(
                        "{} only supports mono audio, not multichannel audio".format(
                            self.__class__.__name__
                        )
                    )
            return self.apply(samples, sample_rate)
        return samples

    def randomize_parameters(self, samples, sample_rate):
        self.parameters["should_apply"] = random.random() < self.p

    def apply(self, samples, sample_rate):
        raise NotImplementedError

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
