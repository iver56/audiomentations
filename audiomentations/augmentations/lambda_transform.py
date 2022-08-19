from typing import Callable

from audiomentations.core.transforms_interface import BaseWaveformTransform


class Lambda(BaseWaveformTransform):
    """
    Apply any choice of operation over the signal at user discretion.
    """

    supports_multichannel = True

    def __init__(self, operator: Callable, p: float = 0.5, **kwargs: dict):
        """
        :param operator: A callable to be applied over samples
        :param p: The probability of applying this transform
        :param **kwargs: The parameters and the values to be passed to the operator.
        """
        super().__init__(p=p)
        self.operator = operator
        self.kwargs = kwargs

    def apply(self, samples, sample_rate):
        return self.operator(samples, sample_rate, **self.kwargs)
