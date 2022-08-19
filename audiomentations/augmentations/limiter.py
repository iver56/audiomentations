import random
import math
import sys

import numpy as np

from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import convert_decibels_to_amplitude_ratio


class Limiter(BaseWaveformTransform):
    """
    A simple audio limiter (dynamic range compression).
    Note: This transform also delays the signal by a fraction of the attack time.
    """

    supports_multichannel = True

    def __init__(
        self,
        min_threshold_db: float = -24,
        max_threshold_db: float = -2,
        min_attack: float = 0.0005,
        max_attack: float = 0.025,
        min_release: float = 0.05,
        max_release: float = 0.7,
        threshold_mode: str = "relative_to_signal_peak",
        p: float = 0.5,
    ):
        """
        The threshold determines the level above which the limiter kicks in.
        The attack time is how quickly the limiter will kick in once the limiting decibel.
        The release time determines how quickly the limiter stops working after the signal drops below the threshold.

        :param min_threshold_db: Min threshold in decibels
        :param max_threshold_db: Max threshold in decibels
        :param min_attack: Minimum attack time in seconds
        :param max_attack: Maximum attack time in seconds
        :param min_release: Minimum release time in seconds
        :param max_release: Minimum release time in seconds
        :param threshold_mode: "relative_to_signal_peak" or "absolute"
            "relative_to_signal_peak" means the threshold is relative to peak of the signal.
            "absolute" means the threshold is relative to 0 dBFS, so it doesn't depend
            on the peak of the signal.
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        assert min_attack > 0.0, "min_attack must be a positive number"
        assert max_attack > 0.0, "max_attack must be a positive number"
        assert (
            max_attack >= min_attack
        ), "max_attack must be greater than or equal to min_attack"
        assert min_release > 0.0, "min_release must be a positive number"
        assert max_release > 0.0, "max_release must be a positive number"
        assert (
            max_release >= min_release
        ), "max_release must be greater than or equal to min_release"
        assert (
            max_threshold_db >= min_threshold_db
        ), "max_threshold_db must be greater than or equal to min_threshold_db"
        assert threshold_mode in (
            "relative_to_signal_peak",
            "absolute",
        ), 'threshold_mode must be either "relative_to_signal_peak" or "absolute"'
        self.min_threshold_db = min_threshold_db
        self.max_threshold_db = max_threshold_db
        self.min_attack = min_attack
        self.max_attack = max_attack
        self.min_release = min_release
        self.max_release = max_release
        self.threshold_mode = threshold_mode

    @staticmethod
    def convert_time_to_coefficient(
        t: float, sample_rate: int, decay_threshold: float = None
    ) -> float:
        if decay_threshold is None:
            # Attack time and release time in this transform are defined as how long
            # it takes to step 1-decay_threshold of the way to a constant target gain.
            # The default threshold used here is inspired by RT60.
            decay_threshold = convert_decibels_to_amplitude_ratio(-60)
        return 10 ** (math.log10(decay_threshold) / max(sample_rate * t, 1.0))

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)

        if self.parameters["should_apply"]:
            attack_seconds = random.uniform(self.min_attack, self.max_attack)
            self.parameters["attack"] = self.convert_time_to_coefficient(
                attack_seconds, sample_rate
            )
            release_seconds = random.uniform(self.min_release, self.max_release)
            self.parameters["release"] = self.convert_time_to_coefficient(
                release_seconds, sample_rate
            )
            # Delay the signal by 60% of the attack time by default
            self.parameters["delay"] = max(round(0.6 * attack_seconds * sample_rate), 1)

            threshold_factor = (
                np.amax(np.abs(samples))
                if self.threshold_mode == "relative_to_signal_peak"
                else 1.0
            )
            threshold_db = random.uniform(self.min_threshold_db, self.max_threshold_db)

            self.parameters[
                "threshold"
            ] = threshold_factor * convert_decibels_to_amplitude_ratio(threshold_db)

    def apply(self, samples, sample_rate):
        try:
            from cylimiter import Limiter as CyLimiter
        except ImportError:
            print(
                "Failed to import cylimiter. Maybe it is not installed? "
                "To install the optional cylimiter dependency of audiomentations,"
                " run `pip install cylimiter` or `pip install audiomentations[extras]`",
                file=sys.stderr,
            )
            raise

        limiter = CyLimiter(
            attack=self.parameters["attack"],
            release=self.parameters["release"],
            delay=self.parameters["delay"],
            threshold=self.parameters["threshold"],
        )
        if samples.ndim == 1:
            processed_samples = np.copy(samples)
            limiter.limit_inplace(processed_samples)
        else:
            # By default, there is no interchannel linking. The channels are processed
            # independently. Support for linking may be added in the future:
            # https://github.com/pzelasko/cylimiter/issues/4
            processed_samples = np.copy(samples)
            for chn_idx in range(samples.shape[0]):
                limiter.reset()
                channel = np.ascontiguousarray(processed_samples[chn_idx, :])
                limiter.limit_inplace(channel)
                processed_samples[chn_idx, :] = channel

        return processed_samples
