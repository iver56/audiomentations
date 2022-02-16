from audiomentations.augmentations.base_butterword_filter import BaseButterworthFilter


class BandPassFilter(BaseButterworthFilter):
    """
    Apply band-pass filtering to the input audio. Filter steepness (6/12/18... dB / octave)
    is parametrized. Can also be set for zero-phase filtering (will result in a 6db drop at
    cutoffs).
    """

    supports_multichannel = True

    def __init__(
        self,
        min_center_freq=200.0,
        max_center_freq=4000.0,
        min_bandwidth_fraction=0.5,
        max_bandwidth_fraction=1.99,
        min_rolloff=12,
        max_rolloff=24,
        zero_phase=False,
        p=0.5,
    ):
        """
        :param min_center_freq: Minimum center frequency in hertz
        :param max_center_freq: Maximum center frequency in hertz
        :param min_bandwidth_fraction: Minimum bandwidth relative to center frequency
        :param max_bandwidth_fraction: Maximum bandwidth relative to center frequency
        :param min_rolloff: Minimum filter roll-off (in db/octave).
            Must be a multiple of 6
        :param max_rolloff: Maximum filter roll-off (in db/octave)
            Must be a multiple of 6
        :param zero_phase: Whether filtering should be zero phase.
            When this is set to `true` it will not affect the phase of the
            input signal but will sound 3db lower at the cutoff frequency
            compared to the non-zero phase case (6db vs 3db). Additionally,
            it is 2X times slower than in the non-zero phase case. If you
            absolutely want no phase distortions (e.g. want to augment an
            audio file with lots of transients, like a drum track), set
            this to `true`.
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
            filter_type="bandpass",
        )
