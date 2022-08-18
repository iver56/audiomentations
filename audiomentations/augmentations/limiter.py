import warnings

import math
import random

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
        :param threshold_mode: "relative_to_signal_peak" or "relative_to_0dbfs"
            "relative_to_signal_peak" means the threshold is relative to peak of the signal
            "relative_to_0dbfs means" the threshold is "absolute" (relative to 0 dbfs),
            so it doesn't depend on the peak of the signal
        :param p: The probability of applying this transform
        """
        super().__init__(p)
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
                1.0
                if self.threshold_mode == "relative_to_0dbfs"
                else np.amax(np.abs(samples))
            )
            threshold_db = random.uniform(self.min_threshold_db, self.max_threshold_db)

            self.parameters[
                "threshold"
            ] = threshold_factor * convert_decibels_to_amplitude_ratio(threshold_db)
            if self.parameters["threshold"] > 1.0:
                warnings.warn(
                    "The input audio has a peak outside the [-1, 1] range."
                    " A threshold above 1 is not supported, so it will be set to a value just below 1."
                    " Normalize your audio before passing it to the limiter to avoid this issue"
                )
                self.parameters["threshold"] = 0.9999999

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
            processed_samples = np.array(limiter.limit(samples), dtype=np.float32)
        else:
            # By default, there is no interchannel linking. The channels are processed
            # independently. Support for linking may be added in the future:
            # https://github.com/pzelasko/cylimiter/issues/4
            processed_samples = np.zeros_like(samples, dtype=np.float32)
            for chn_idx in range(samples.shape[0]):
                limiter.reset()
                processed_samples[chn_idx, :] = np.array(
                    limiter.limit(samples[chn_idx, :]), dtype=np.float32
                )

        return processed_samples
