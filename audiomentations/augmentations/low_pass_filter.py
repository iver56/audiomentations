from audiomentations.augmentations.base_butterword_filter import BaseButterworthFilter


class LowPassFilter(BaseButterworthFilter):
    """
    Apply low-pass filtering to the input audio of parametrized filter steepness (6/12/18... dB / octave).
    Can also be set for zero-phase filtering (will result in a 6db drop at cutoff).
    """

    supports_multichannel = True

    def __init__(
        self,
        min_cutoff_freq=150,
        max_cutoff_freq=7500,
        min_rolloff=12,
        max_rolloff=24,
        zero_phase=False,
        p: float = 0.5,
    ):
        """
        :param min_cutoff_freq: Minimum cutoff frequency in hertz
        :param max_cutoff_freq: Maximum cutoff frequency in hertz
        :param min_rolloff: Minimum filter roll-off (in db/octave).
            Must be a multiple of 6
        :param max_rolloff: Maximum filter roll-off (in db/octave)
            Must be a multiple of 6
        :param zero_phase: Whether filtering should be zero phase.
            When this is set to `true` it will not affect the phase of the
            input signal but will sound 3db lower at the cutoff frequency
            compared to the non-zero phase case (6db vs 3db). Additionally,
            it is 2X times slower than in the non-zero phase case. If you
            absolutely want no phase distortions (e.g. want to augment a
            drum track), set this to `true`.
        :param p: The probability of applying this transform
        """
        super().__init__(
            min_cutoff_freq=min_cutoff_freq,
            max_cutoff_freq=max_cutoff_freq,
            min_rolloff=min_rolloff,
            max_rolloff=max_rolloff,
            zero_phase=zero_phase,
            p=p,
            filter_type="lowpass",
        )
