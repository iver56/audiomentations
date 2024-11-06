import random

from audiomentations.core.transforms_interface import BaseSpectrogramTransform


class SpecChannelShuffle(BaseSpectrogramTransform):
    """
    Shuffle the channels of a multichannel spectrogram (channels last).
    This can help combat positional bias.
    """
    supports_multichannel = True
    supports_mono = False

    def randomize_parameters(self, magnitude_spectrogram):
        super().randomize_parameters(magnitude_spectrogram)
        if self.parameters["should_apply"]:
            self.parameters["shuffled_channel_indexes"] = list(range(magnitude_spectrogram.shape[-1]))
            random.shuffle(self.parameters["shuffled_channel_indexes"])

    def apply(self, magnitude_spectrogram):
        return magnitude_spectrogram[..., self.parameters["shuffled_channel_indexes"]]
