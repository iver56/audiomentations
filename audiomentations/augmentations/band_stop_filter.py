from audiomentations.augmentations.base_butterword_filter import BaseButterworthFilter


class BandStopFilter(BaseButterworthFilter):
    """
    Apply band-stop filtering to the input audio. Also known as notch filter or
    band reject filter. It relates to the frequency mask idea in the SpecAugment paper.
    Center frequency gets picked in mel space, so it is
    more aligned with human hearing, which is not linear. Filter steepness
    (6/12/18... dB / octave) is parametrized. Can also be set for zero-phase filtering
    (will result in a 6 dB drop at cutoffs).
    """

    supports_multichannel = True

    def __init__(
        self,
        min_center_freq: float = 200.0,
        max_center_freq: float = 4000.0,
        min_bandwidth_fraction: float = 0.5,
        max_bandwidth_fraction: float = 1.99,
        min_rolloff: int = 12,
        max_rolloff: int = 24,
        zero_phase: bool = False,
        p: float = 0.5,
    ):
        """
        :param min_center_freq: Minimum center frequency in hertz
        :param max_center_freq: Maximum center frequency in hertz
        :param min_bandwidth_fraction: Minimum bandwidth fraction relative to center
            frequency (number between 0 and 2)
        :param max_bandwidth_fraction: Maximum bandwidth fraction relative to center
            frequency (number between 0 and 2)
        :param min_rolloff: Minimum filter roll-off (in dB/octave).
            Must be a multiple of 6
        :param max_rolloff: Maximum filter roll-off (in dB/octave)
            Must be a multiple of 6
        :param zero_phase: Whether filtering should be zero phase.
            When this is set to `True`, it will not affect the phase of the
            input signal but will sound 3 dB lower at the cutoff frequency
            compared to the non-zero phase case (6 dB vs 3 dB). Additionally,
            it is twice as slow as the non-zero phase case. If you
            absolutely want no phase distortions (e.g. want to augment a
            drum track), set this to `True`.
        :param p: The probability of applying this transform
        """
        super().__init__(
            min_center_freq=min_center_freq,
            max_center_freq=max_center_freq,
            min_bandwidth_fraction=min_bandwidth_fraction,
            max_bandwidth_fraction=max_bandwidth_fraction,
            min_rolloff=min_rolloff,
            max_rolloff=max_rolloff,
            zero_phase=zero_phase,
            p=p,
            filter_type="bandstop",
        )
