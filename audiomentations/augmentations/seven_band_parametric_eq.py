import numpy as np
from numpy.typing import NDArray

from audiomentations import LowShelfFilter, PeakingFilter, HighShelfFilter
from audiomentations.core.transforms_interface import BaseWaveformTransform


class SevenBandParametricEQ(BaseWaveformTransform):
    """
    Adjust the volume of different frequency bands. This transform is a 7-band
    parametric equalizer - a combination of one low shelf filter, five peaking filters
    and one high shelf filter, all with randomized gains, Q values and center frequencies.

    Because this transform changes the timbre but keeps the overall "class" of the
    sound the same (depending on the application), it can be used for data augmentation to
    make ML models more robust to various frequency spectra. Many things can affect
    the spectrum, for example:

    * the nature and quality of the sound source
    * room acoustics
    * any objects between the microphone and the sound source
    * microphone type or model
    * the distance between the sound source and the microphone

    The seven bands have center frequencies picked in the following ranges (min-max):

    * 42-95 Hz
    * 91-204 Hz
    * 196-441 Hz
    * 421-948 Hz
    * 909-2045 Hz
    * 1957-4404 Hz
    * 4216-9486 Hz
    """

    supports_multichannel = True

    def __init__(
        self,
        min_gain_db: float = -12.0,
        max_gain_db: float = 12.0,
        p: float = 0.5,
    ):
        """
        :param min_gain_db: Minimum number of dB to cut or boost a band
        :param max_gain_db: Maximum number of dB to cut or boost a band
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        assert min_gain_db <= max_gain_db
        
        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db

        self.low_shelf_filter = LowShelfFilter(
            min_center_freq=42.0,
            max_center_freq=95.0,
            min_gain_db=min_gain_db,
            max_gain_db=max_gain_db,
            p=1.0,
        )
        self.peaking_filters = [
            PeakingFilter(
                min_center_freq=91.0,
                max_center_freq=204.0,
                min_gain_db=min_gain_db,
                max_gain_db=max_gain_db,
                p=1.0,
            ),
            PeakingFilter(
                min_center_freq=196.0,
                max_center_freq=441.0,
                min_gain_db=min_gain_db,
                max_gain_db=max_gain_db,
                p=1.0,
            ),
            PeakingFilter(
                min_center_freq=421.0,
                max_center_freq=948.0,
                min_gain_db=min_gain_db,
                max_gain_db=max_gain_db,
                p=1.0,
            ),
            PeakingFilter(
                min_center_freq=909.0,
                max_center_freq=2045.0,
                min_gain_db=min_gain_db,
                max_gain_db=max_gain_db,
                p=1.0,
            ),
            PeakingFilter(
                min_center_freq=1957.0,
                max_center_freq=4404.0,
                min_gain_db=min_gain_db,
                max_gain_db=max_gain_db,
                p=1.0,
            ),
        ]
        self.high_shelf_filter = HighShelfFilter(
            min_center_freq=4216.0,
            max_center_freq=9486.0,
            min_gain_db=min_gain_db,
            max_gain_db=max_gain_db,
            p=1.0,
        )
        self.low_shelf_filter.freeze_parameters()
        for i in range(len(self.peaking_filters)):
            self.peaking_filters[i].freeze_parameters()
        self.high_shelf_filter.freeze_parameters()

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        self.low_shelf_filter.randomize_parameters(samples, sample_rate)
        for i in range(len(self.peaking_filters)):
            self.peaking_filters[i].randomize_parameters(samples, sample_rate)
        self.high_shelf_filter.randomize_parameters(samples, sample_rate)

    def apply(self, samples: NDArray[np.float32], sample_rate: int) -> NDArray[np.float32]:
        samples = self.low_shelf_filter(samples, sample_rate)
        for i in range(len(self.peaking_filters)):
            samples = self.peaking_filters[i](samples, sample_rate)
        samples = self.high_shelf_filter(samples, sample_rate)
        return samples
