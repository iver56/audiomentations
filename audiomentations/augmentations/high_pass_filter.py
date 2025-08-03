from audiomentations.augmentations.base_butterword_filter import BaseButterworthFilter


class HighPassFilter(BaseButterworthFilter):
    """
    Apply high-pass filtering to the input audio of parametrized filter steepness (6/12/18... dB / octave).
    Can also be set for zero-phase filtering (will result in a 6 dB drop at cutoff).
    """

    supports_multichannel = True

    def __init__(
        self,
        min_cutoff_freq: float = 20.0,
        max_cutoff_freq: float = 2400.0,
        min_rolloff: int = 12,
        max_rolloff: int = 24,
        zero_phase: bool = False,
        p: float = 0.5,
    ):
        """
        :param min_cutoff_freq: Minimum cutoff frequency in hertz
        :param max_cutoff_freq: Maximum cutoff frequency in hertz
        :param min_rolloff: Minimum filter roll-off (in dB/octave).
            Must be a multiple of 6
        :param max_rolloff: Maximum filter roll-off (in dB/octave)
            Must be a multiple of 6
        :param zero_phase: Whether filtering should be zero phase.
            When this is set to `True`, it will not affect the phase of the
            input signal but will sound 3 dB lower at the cutoff frequency
            compared to the non-zero phase case (6 dB vs. 3 dB). Additionally,
            it is twice as slow as the non-zero phase case. If you
            absolutely want no phase distortions (e.g. want to augment a
            drum track), set this to `True`.
        :param p: The probability of applying this transform
        """
        if min_cutoff_freq <= 0:
            raise ValueError(
                f"HighPassFilter requires min_cutoff_freq > 0. Got {min_cutoff_freq}."
            )
        super().__init__(
            min_cutoff_freq=min_cutoff_freq,
            max_cutoff_freq=max_cutoff_freq,
            min_rolloff=min_rolloff,
            max_rolloff=max_rolloff,
            zero_phase=zero_phase,
            p=p,
            filter_type="highpass",
        )
