from audiomentations import LowShelfFilter, PeakingFilter, HighShelfFilter
from audiomentations.core.transforms_interface import BaseWaveformTransform


class Equalizer(BaseWaveformTransform):
    """
    Adjust the volume of different frequency bands. This transform is a 7-band
    parametric equalizer - a combination of one low shelf filter, five peaking filters
    and one high shelf filter, all with randomized gains, q values and center frequencies.
    """

    supports_multichannel = True

    def __init__(
        self,
        min_gain_in_db: float = -12.0,
        max_gain_in_db: float = 12.0,
        p: float = 0.5,
    ):
        """
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        assert min_gain_in_db <= max_gain_in_db

        self.low_shelf_filter = LowShelfFilter(
            min_center_freq=63.0 / 1.5,
            max_center_freq=63.0 * 1.5,
            min_gain_db=min_gain_in_db,
            max_gain_db=max_gain_in_db,
            p=1.0,
        )
        self.peaking_filters = [
            PeakingFilter(
                min_center_freq=136.0 / 1.5,
                max_center_freq=136.0 * 1.5,
                min_gain_db=min_gain_in_db,
                max_gain_db=max_gain_in_db,
                p=1.0,
            ),
            PeakingFilter(
                min_center_freq=294.0 / 1.5,
                max_center_freq=294.0 * 1.5,
                min_gain_db=min_gain_in_db,
                max_gain_db=max_gain_in_db,
                p=1.0,
            ),
            PeakingFilter(
                min_center_freq=632.0 / 1.5,
                max_center_freq=632.0 * 1.5,
                min_gain_db=min_gain_in_db,
                max_gain_db=max_gain_in_db,
                p=1.0,
            ),
            PeakingFilter(
                min_center_freq=1363.0 / 1.5,
                max_center_freq=1363.0 * 1.5,
                min_gain_db=min_gain_in_db,
                max_gain_db=max_gain_in_db,
                p=1.0,
            ),
            PeakingFilter(
                min_center_freq=2936.0 / 1.5,
                max_center_freq=2936.0 * 1.5,
                min_gain_db=min_gain_in_db,
                max_gain_db=max_gain_in_db,
                p=1.0,
            ),
        ]
        self.high_shelf_filter = HighShelfFilter(
            min_center_freq=6324.0 / 1.5,
            max_center_freq=6324.0 * 1.5,
            min_gain_db=min_gain_in_db,
            max_gain_db=max_gain_in_db,
            p=1.0,
        )
        self.low_shelf_filter.freeze_parameters()
        for i in range(len(self.peaking_filters)):
            self.peaking_filters[i].freeze_parameters()
        self.high_shelf_filter.freeze_parameters()

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        self.low_shelf_filter.randomize_parameters(samples, sample_rate)
        for i in range(len(self.peaking_filters)):
            self.peaking_filters[i].randomize_parameters(samples, sample_rate)
        self.high_shelf_filter.randomize_parameters(samples, sample_rate)

    def apply(self, samples, sample_rate):
        samples = self.low_shelf_filter(samples, sample_rate)
        for i in range(len(self.peaking_filters)):
            samples = self.peaking_filters[i](samples, sample_rate)
        samples = self.high_shelf_filter(samples, sample_rate)
        return samples
