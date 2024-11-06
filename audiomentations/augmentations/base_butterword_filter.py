import random

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfilt, sosfiltfilt, sosfilt_zi

from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import (
    convert_frequency_to_mel,
    convert_mel_to_frequency,
)


class BaseButterworthFilter(BaseWaveformTransform):
    """
    A `scipy.signal.butter`-based generic filter class.
    """

    supports_multichannel = True

    # The types below must be equal to the ones accepted by
    # the `btype` argument of `scipy.signal.butter`
    ALLOWED_ONE_SIDE_FILTER_TYPES = ("lowpass", "highpass")
    ALLOWED_TWO_SIDE_FILTER_TYPES = ("bandpass", "bandstop")
    ALLOWED_FILTER_TYPES = ALLOWED_ONE_SIDE_FILTER_TYPES + ALLOWED_TWO_SIDE_FILTER_TYPES

    def __init__(self, **kwargs):
        assert "p" in kwargs
        assert "min_rolloff" in kwargs
        assert "max_rolloff" in kwargs
        assert "filter_type" in kwargs
        assert "zero_phase" in kwargs

        super().__init__(kwargs["p"])

        self.filter_type = kwargs["filter_type"]
        self.min_rolloff = kwargs["min_rolloff"]
        self.max_rolloff = kwargs["max_rolloff"]
        self.zero_phase = kwargs["zero_phase"]

        if self.zero_phase:
            assert (
                self.min_rolloff % 12 == 0
            ), "Zero phase filters can only have a steepness which is a multiple of 12 dB/octave"
            assert (
                self.max_rolloff % 12 == 0
            ), "Zero phase filters can only have a steepness which is a multiple of 12 dB/octave"
        else:
            assert (
                self.min_rolloff % 6 == 0
            ), "Non zero phase filters can only have a steepness which is a multiple of 6 dB/octave"
            assert (
                self.max_rolloff % 6 == 0
            ), "Non zero phase filters can only have a steepness which is a multiple of 6 dB/octave"

        assert (
            self.filter_type in BaseButterworthFilter.ALLOWED_FILTER_TYPES
        ), "Filter type must be one of: " + ", ".join(
            BaseButterworthFilter.ALLOWED_FILTER_TYPES
        )

        assert ("min_cutoff_freq" in kwargs and "max_cutoff_freq" in kwargs) or (
            "min_center_freq" in kwargs
            and "max_center_freq" in kwargs
            and "min_bandwidth_fraction" in kwargs
            and "max_bandwidth_fraction" in kwargs
        ), "Arguments for either a one-sided, or a two-sided filter must be given"

        if "min_cutoff_freq" in kwargs:
            self.initialize_one_sided_filter(
                min_cutoff_freq=kwargs["min_cutoff_freq"],
                max_cutoff_freq=kwargs["max_cutoff_freq"],
            )
        elif "min_center_freq" in kwargs:
            self.initialize_two_sided_filter(
                min_center_freq=kwargs["min_center_freq"],
                max_center_freq=kwargs["max_center_freq"],
                min_bandwidth_fraction=kwargs["min_bandwidth_fraction"],
                max_bandwidth_fraction=kwargs["max_bandwidth_fraction"],
            )

    def initialize_one_sided_filter(
        self,
        min_cutoff_freq,
        max_cutoff_freq,
    ):
        """
        :param min_cutoff_freq: Minimum cutoff frequency in hertz
        :param max_cutoff_freq: Maximum cutoff frequency in hertz
        :param min_rolloff: Minimum filter roll-off (in dB/octave).
            Must be a multiple of 6
        :param max_rolloff: Maximum filter roll-off (in dB/octave)
            Must be a multiple of 6
        :param p: The probability of applying this transform
        """

        self.min_cutoff_freq = min_cutoff_freq
        self.max_cutoff_freq = max_cutoff_freq
        if self.min_cutoff_freq > self.max_cutoff_freq:
            raise ValueError("min_cutoff_freq must not be greater than max_cutoff_freq")

        if self.min_rolloff < 6 or self.min_rolloff % 6 != 0:
            raise ValueError(
                "min_rolloff must be 6 or greater, as well as a multiple of 6 (e.g. 6, 12, 18, 24...)"
            )
        if self.max_rolloff < 6 or self.max_rolloff % 6 != 0:
            raise ValueError(
                "max_rolloff must be 6 or greater, as well as a multiple of 6 (e.g. 6, 12, 18, 24...)"
            )
        if self.min_rolloff > self.max_rolloff:
            raise ValueError("min_rolloff must not be greater than max_rolloff")

    def initialize_two_sided_filter(
        self,
        min_center_freq,
        max_center_freq,
        min_bandwidth_fraction,
        max_bandwidth_fraction,
    ):
        """
        :param min_center_freq: Minimum center frequency in hertz
        :param max_center_freq: Maximum center frequency in hertz
        :param min_bandwidth_fraction: Minimum bandwidth fraction relative to center
            frequency (number between 0.0 and 2.0)
        :param max_bandwidth_fraction: Maximum bandwidth fraction relative to center
            frequency (number between 0.0 and 2.0)
        :param min_rolloff: Minimum filter roll-off (in dB/octave).
            Must be a multiple of 6
        :param max_rolloff: Maximum filter roll-off (in dB/octave)
            Must be a multiple of 6
        :param p: The probability of applying this transform
        """

        self.min_center_freq = min_center_freq
        self.max_center_freq = max_center_freq
        self.min_bandwidth_fraction = min_bandwidth_fraction
        self.max_bandwidth_fraction = max_bandwidth_fraction

        if self.min_center_freq > self.max_center_freq:
            raise ValueError("min_center_freq must not be greater than max_center_freq")
        if self.min_bandwidth_fraction <= 0.0:
            raise ValueError("min_bandwidth_fraction must be a positive number")
        if self.max_bandwidth_fraction >= 2.0:
            raise ValueError(
                "max_bandwidth_fraction should be smaller than 2.0, since otherwise"
                " the low cut frequency of the band can be smaller than 0 Hz."
            )
        if self.min_bandwidth_fraction > self.max_bandwidth_fraction:
            raise ValueError(
                "min_bandwidth_fraction must not be greater than max_bandwidth_fraction"
            )

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int = None):
        super().randomize_parameters(samples, sample_rate)
        if self.zero_phase:
            random_order = random.randint(
                self.min_rolloff // 12, self.max_rolloff // 12
            )
            self.parameters["rolloff"] = random_order * 12
        else:
            random_order = random.randint(self.min_rolloff // 6, self.max_rolloff // 6)
            self.parameters["rolloff"] = random_order * 6

        if self.filter_type in BaseButterworthFilter.ALLOWED_ONE_SIDE_FILTER_TYPES:
            cutoff_mel = np.random.uniform(
                low=convert_frequency_to_mel(self.min_cutoff_freq),
                high=convert_frequency_to_mel(self.max_cutoff_freq),
            )
            self.parameters["cutoff_freq"] = convert_mel_to_frequency(cutoff_mel)
        elif self.filter_type in BaseButterworthFilter.ALLOWED_TWO_SIDE_FILTER_TYPES:
            center_mel = np.random.uniform(
                low=convert_frequency_to_mel(self.min_center_freq),
                high=convert_frequency_to_mel(self.max_center_freq),
            )
            self.parameters["center_freq"] = convert_mel_to_frequency(center_mel)

            bandwidth_fraction = np.random.uniform(
                low=self.min_bandwidth_fraction, high=self.max_bandwidth_fraction
            )
            self.parameters["bandwidth"] = (
                self.parameters["center_freq"] * bandwidth_fraction
            )

    def apply(self, samples: NDArray[np.float32], sample_rate: int = None) -> NDArray[np.float32]:
        assert samples.dtype == np.float32

        if self.filter_type in BaseButterworthFilter.ALLOWED_ONE_SIDE_FILTER_TYPES:
            cutoff_freq = self.parameters["cutoff_freq"]
            nyquist_freq = sample_rate // 2
            if cutoff_freq > nyquist_freq:
                # Ensure that the cutoff frequency does not exceed the Nyquist
                # frequency to avoid an exception from scipy
                cutoff_freq = nyquist_freq * 0.9999
            sos = butter(
                self.parameters["rolloff"] // (12 if self.zero_phase else 6),
                cutoff_freq,
                btype=self.filter_type,
                analog=False,
                fs=sample_rate,
                output="sos",
            )
        elif self.filter_type in BaseButterworthFilter.ALLOWED_TWO_SIDE_FILTER_TYPES:
            low_freq = self.parameters["center_freq"] - self.parameters["bandwidth"] / 2
            high_freq = (
                self.parameters["center_freq"] + self.parameters["bandwidth"] / 2
            )
            nyquist_freq = sample_rate // 2
            if high_freq > nyquist_freq:
                # Ensure that the upper critical frequency does not exceed the Nyquist
                # frequency to avoid an exception from scipy
                high_freq = nyquist_freq * 0.9999
            sos = butter(
                self.parameters["rolloff"] // (12 if self.zero_phase else 6),
                [low_freq, high_freq],
                btype=self.filter_type,
                analog=False,
                fs=sample_rate,
                output="sos",
            )

        # The actual processing takes place here
        if len(samples.shape) == 1:
            if self.zero_phase:
                processed_samples = sosfiltfilt(sos, samples)
            else:
                processed_samples, _ = sosfilt(
                    sos, samples, zi=sosfilt_zi(sos) * samples[0]
                )
            processed_samples = processed_samples.astype(np.float32)
        else:
            processed_samples = np.zeros_like(samples, dtype=np.float32)
            if self.zero_phase:
                for chn_idx in range(samples.shape[0]):
                    processed_samples[chn_idx, :] = sosfiltfilt(
                        sos, samples[chn_idx, :]
                    )
            else:
                zi = sosfilt_zi(sos)
                for chn_idx in range(samples.shape[0]):
                    processed_samples[chn_idx, :], _ = sosfilt(
                        sos, samples[chn_idx, :], zi=zi * samples[chn_idx, 0]
                    )

        return processed_samples
