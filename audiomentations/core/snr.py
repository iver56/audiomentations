import math


class SNR(object):
    """A way of representing Signal-to-Noise Ratio."""

    def __init__(self, decibels=None, amplitude_ratio=None):
        if decibels is None and amplitude_ratio is None:
            raise Exception(
                "When instantiating an SNR instance, you need to specify either decibel or"
                " amplitude_ratio"
            )
        if decibels is not None and amplitude_ratio is not None:
            raise Exception(
                "You cannot specify both decibel and amplitude_ratio. Please specify only one"
                " of them."
            )

        if decibels is not None:
            self._decibels = decibels
        elif amplitude_ratio is not None:
            self._decibels = self.convert_amplitude_ratio_to_decibels(amplitude_ratio)

    @staticmethod
    def convert_decibels_to_amplitude_ratio(decibels):
        return 10 ** (decibels / 20)

    @staticmethod
    def convert_amplitude_ratio_to_decibels(amplitude_ratio):
        return 20 * math.log10(amplitude_ratio)

    @property
    def decibels(self):
        return self._decibels

    @decibels.setter
    def decibels(self, value):
        self._decibels = value

    @property
    def amplitude_ratio(self):
        return self.convert_decibels_to_amplitude_ratio(self._decibels)

    @amplitude_ratio.setter
    def amplitude_ratio(self, value):
        self._decibels = self.convert_amplitude_ratio_to_decibels(value)
