import random
import warnings

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt, sosfilt_zi

from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import (
    convert_frequency_to_mel,
    convert_mel_to_frequency,
)


class ButterworthFilter(BaseWaveformTransform):
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
            ), "Zero phase filters can only have a steepness which is a multiple of 12db/octave"
            assert (
                self.max_rolloff % 12 == 0
            ), "Zero phase filters can only have a steepness which is a multiple of 12db/octave"
        else:
            assert (
                self.min_rolloff % 6 == 0
            ), "Non zero phase filters can only have a steepness which is a multiple of 6db/octave"
            assert (
                self.max_rolloff % 6 == 0
            ), "Non zero phase filters can only have a steepness which is a multiple of 6db/octave"

        assert (
            self.filter_type in ButterworthFilter.ALLOWED_FILTER_TYPES
        ), "Filter type must be one of: " + ", ".join(
            ButterworthFilter.ALLOWED_FILTER_TYPES
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
        :param min_rolloff: Minimum filter roll-off (in db/octave).
            Must be a multiple of 6
        :param max_rolloff: Maximum filter roll-off (in db/octave)
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
        :param min_rolloff: Minimum filter roll-off (in db/octave).
            Must be a multiple of 6
        :param max_rolloff: Maximum filter roll-off (in db/octave)
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

    def randomize_parameters(self, samples: np.array, sample_rate: int = None):
        super().randomize_parameters(samples, sample_rate)
        if self.zero_phase:
            random_order = random.randint(
                self.min_rolloff // 12, self.max_rolloff // 12
            )
            self.parameters["rolloff"] = random_order * 12
        else:
            random_order = random.randint(self.min_rolloff // 6, self.max_rolloff // 6)
            self.parameters["rolloff"] = random_order * 6

        if self.filter_type in ButterworthFilter.ALLOWED_ONE_SIDE_FILTER_TYPES:
            cutoff_mel = np.random.uniform(
                low=convert_frequency_to_mel(self.min_cutoff_freq),
                high=convert_frequency_to_mel(self.max_cutoff_freq),
            )
            self.parameters["cutoff_freq"] = convert_mel_to_frequency(cutoff_mel)
        elif self.filter_type in ButterworthFilter.ALLOWED_TWO_SIDE_FILTER_TYPES:
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

    def apply(self, samples: np.array, sample_rate: int = None):
        assert samples.dtype == np.float32

        if self.filter_type in ButterworthFilter.ALLOWED_ONE_SIDE_FILTER_TYPES:
            cutoff_freq = self.parameters["cutoff_freq"]
            nyquist_freq = sample_rate // 2
            if cutoff_freq > nyquist_freq:
                # Ensure that the cutoff frequency does not exceed the nyquist
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
        elif self.filter_type in ButterworthFilter.ALLOWED_TWO_SIDE_FILTER_TYPES:
            low_freq = self.parameters["center_freq"] - self.parameters["bandwidth"] / 2
            high_freq = (
                self.parameters["center_freq"] + self.parameters["bandwidth"] / 2
            )
            nyquist_freq = sample_rate // 2
            if high_freq > nyquist_freq:
                # Ensure that the upper critical frequency does not exceed the nyquist
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


class LowPassFilter(ButterworthFilter):
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


class HighPassFilter(ButterworthFilter):
    """
    Apply high-pass filtering to the input audio of parametrized filter steepness (6/12/18... dB / octave).
    Can also be set for zero-phase filtering (will result in a 6db drop at cutoff).
    """

    supports_multichannel = True

    def __init__(
        self,
        min_cutoff_freq=20,
        max_cutoff_freq=2400,
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
            filter_type="highpass",
        )


class BandStopFilter(ButterworthFilter):
    """
    Apply band-stop filtering to the input audio. Also known as notch filter or
    band reject filter. It relates to the frequency mask idea in the SpecAugment paper.
    This transform is similar to FrequencyMask, but has overhauled default parameters
    and parameter randomization - center frequency gets picked in mel space so it is
    more aligned with human hearing, which is not linear. Filter steepness
    (6/12/18... dB / octave) is parametrized. Can also be set for zero-phase filtering
    (will result in a 6db drop at cutoffs).
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
        :param min_bandwidth_fraction: Minimum bandwidth fraction relative to center
            frequency (number between 0 and 2)
        :param max_bandwidth_fraction: Maximum bandwidth fraction relative to center
            frequency (number between 0 and 2)
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


class BandPassFilter(ButterworthFilter):
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


class PeakingFilter(BaseWaveformTransform):
    """
    Peaking filter transform. Applies a peaking filter at a specific center frequency in hertz
    of a specific gain in db (note: can be positive or negative!), and a quality factor
    parameter. Filter coefficients are taken from the W3 Audio EQ Cookbook:
    https://www.w3.org/TR/audio-eq-cookbook/
    """

    supports_multichannel = True

    def __init__(
        self,
        min_center_freq=50.0,
        max_center_freq=7500.0,
        min_gain_db=-24,
        max_gain_db=24,
        min_q=0.5,
        max_q=5.0,
        p=0.5,
    ):
        """
        :param min_center_freq: The minimum center frequency of the peaking filter
        :param max_center_freq: The maximum center frequency of the peaking filter
        :param min_gain_db: The minimum gain at center frequency in db
        :param max_gain_db: The maximum gain at center frequency in db
        :param min_q: The minimum quality factor Q. The higher the Q, the steeper the
            transition band will be.
        :param max_q: The maximum quality factor Q. The higher the Q, the steeper the
            transition band will be.
        """

        assert (
            min_center_freq <= max_center_freq
        ), "`min_center_freq` should be no greater than `max_center_freq`"
        assert (
            min_gain_db <= max_gain_db
        ), "`min_gain_db` should be no greater than `max_gain_db`"

        assert min_q > 0, "`min_q` should be greater than 0"
        assert max_q > 0, "`max_q` should be greater than 0"

        super().__init__(p)

        self.min_center_freq = min_center_freq
        self.max_center_freq = max_center_freq

        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db

        self.min_q = min_q
        self.max_q = max_q

    def _get_biquad_coefficients_from_input_parameters(
        self, center_freq, gain_db, q_factor, sample_rate
    ):
        normalized_frequency = 2 * np.pi * center_freq / sample_rate
        gain = 10 ** (gain_db / 40)
        alpha = np.sin(normalized_frequency) / 2 / q_factor

        b0 = 1 + alpha * gain
        b1 = -2 * np.cos(normalized_frequency)
        b2 = 1 - alpha * gain

        a0 = 1 + alpha / gain
        a1 = -2 * np.cos(normalized_frequency)
        a2 = 1 - alpha / gain

        # Return it in `sos` format
        sos = np.array([[b0 / a0, b1 / a0, b2 / a0, 1, a1 / a0, a2 / a0]])

        return sos

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)

        center_mel = np.random.uniform(
            low=convert_frequency_to_mel(self.min_center_freq),
            high=convert_frequency_to_mel(self.max_center_freq),
        )
        self.parameters["center_freq"] = convert_mel_to_frequency(center_mel)
        self.parameters["gain_db"] = random.uniform(self.min_gain_db, self.max_gain_db)
        self.parameters["q_factor"] = random.uniform(self.min_q, self.max_q)

    def apply(self, samples, sample_rate):
        assert samples.dtype == np.float32

        sos = self._get_biquad_coefficients_from_input_parameters(
            self.parameters["center_freq"],
            self.parameters["gain_db"],
            self.parameters["q_factor"],
            sample_rate,
        )

        # The processing takes place here
        zi = sosfilt_zi(sos)
        if len(samples.shape) == 1:
            processed_samples, _ = sosfilt(sos, samples, zi=zi * samples[0])
            processed_samples = processed_samples.astype(np.float32)
        else:
            processed_samples = np.zeros_like(samples, dtype=np.float32)
            for chn_idx in range(samples.shape[0]):
                processed_samples[chn_idx, :], _ = sosfilt(
                    sos, samples[chn_idx, :], zi=zi * samples[chn_idx, 0]
                )

        return processed_samples


class LowShelfFilter(BaseWaveformTransform):
    """
    Low-shelf filter transform. Applies a low-shelf filter at a specific center frequency in hertz.
    The gain at DC frequency is controlled by `{min,max}_gain_db` (note: can be positive or negative!).
    Filter coefficients are taken from the W3 Audio EQ Cookbook: https://www.w3.org/TR/audio-eq-cookbook/
    """

    supports_multichannel = True

    def __init__(
        self,
        min_center_freq=50.0,
        max_center_freq=4000.0,
        min_gain_db=-18.0,
        max_gain_db=18.0,
        min_q=0.1,
        max_q=0.999,
        p=0.5,
    ):

        """
        :param min_center_freq: The minimum center frequency of the shelving filter
        :param max_center_freq: The maximum center frequency of the shelving filter
        :param min_gain_db: The minimum gain at DC (0 hz)
        :param max_gain_db: The maximum gain at DC (0 hz)
        :param min_q: The minimum quality factor q
        :param max_q: The maximum quality factor q
        """

        assert (
            min_center_freq <= max_center_freq
        ), "`min_center_freq` should be no greater than `max_center_freq`"
        assert (
            min_gain_db <= max_gain_db
        ), "`min_gain_db` should be no greater than `max_gain_db`"

        assert 0 < min_q <= 1, "`min_q` should be greater than 0 and less or equal to 1"
        assert 0 < max_q <= 1, "`max_q` should be greater than 0 and less or equal to 1"

        super().__init__(p)

        self.min_center_freq = min_center_freq
        self.max_center_freq = max_center_freq

        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db

        self.min_q = min_q
        self.max_q = max_q

    def _get_biquad_coefficients_from_input_parameters(
        self, center_freq, gain_db, q_factor, sample_rate
    ):
        normalized_frequency = 2 * np.pi * center_freq / sample_rate
        gain = 10 ** (gain_db / 40)
        alpha = np.sin(normalized_frequency) / 2 / q_factor

        b0 = gain * (
            (gain + 1)
            - (gain - 1) * np.cos(normalized_frequency)
            + 2 * np.sqrt(gain) * alpha
        )

        b1 = 2 * gain * ((gain - 1) - (gain + 1) * np.cos(normalized_frequency))

        b2 = gain * (
            (gain + 1)
            - (gain - 1) * np.cos(normalized_frequency)
            - 2 * np.sqrt(gain) * alpha
        )

        a0 = (
            (gain + 1)
            + (gain - 1) * np.cos(normalized_frequency)
            + 2 * np.sqrt(gain) * alpha
        )

        a1 = -2 * ((gain - 1) + (gain + 1) * np.cos(normalized_frequency))

        a2 = (
            (gain + 1)
            + (gain - 1) * np.cos(normalized_frequency)
            - 2 * np.sqrt(gain) * alpha
        )

        # Return it in `sos` format
        sos = np.array([[b0 / a0, b1 / a0, b2 / a0, 1, a1 / a0, a2 / a0]])

        return sos

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)

        center_mel = np.random.uniform(
            low=convert_frequency_to_mel(self.min_center_freq),
            high=convert_frequency_to_mel(self.max_center_freq),
        )
        self.parameters["center_freq"] = convert_mel_to_frequency(center_mel)
        self.parameters["gain_db"] = random.uniform(self.min_gain_db, self.max_gain_db)
        self.parameters["q_factor"] = random.uniform(self.min_q, self.max_q)

    def apply(self, samples, sample_rate):
        assert samples.dtype == np.float32

        sos = self._get_biquad_coefficients_from_input_parameters(
            self.parameters["center_freq"],
            self.parameters["gain_db"],
            self.parameters["q_factor"],
            sample_rate,
        )

        # The processing takes place here
        zi = sosfilt_zi(sos)
        if len(samples.shape) == 1:
            processed_samples, _ = sosfilt(sos, samples, zi=zi * samples[0])
            processed_samples = processed_samples.astype(np.float32)
        else:
            processed_samples = np.zeros_like(samples, dtype=np.float32)
            for chn_idx in range(samples.shape[0]):
                processed_samples[chn_idx, :], _ = sosfilt(
                    sos, samples[chn_idx, :], zi=zi * samples[chn_idx, 0]
                )

        return processed_samples


class HighShelfFilter(BaseWaveformTransform):
    """
    High-shelf filter transform. Applies a high-shelf filter at a specific center frequency in hertz.
    The gain at nyquist frequency is controlled by `{min,max}_gain_db` (note: can be positive or negative!).
    Filter coefficients are taken from the W3 Audio EQ Cookbook: https://www.w3.org/TR/audio-eq-cookbook/
    """

    supports_multichannel = True

    def __init__(
        self,
        min_center_freq=300.0,
        max_center_freq=7500.0,
        min_gain_db=-18.0,
        max_gain_db=18.0,
        min_q=0.1,
        max_q=0.999,
        p=0.5,
    ):
        """
        :param min_center_freq: The minimum center frequency of the shelving filter
        :param max_center_freq: The maximum center frequency of the shelving filter
        :param min_gain_db: The minimum gain at the nyquist frequency
        :param max_gain_db: The maximum gain at the nyquist frequency
        :param min_q: The minimum quality factor Q. The higher the Q, the steeper the
            transition band will be.
        :param max_q: The maximum quality factor Q. The higher the Q, the steeper the
            transition band will be.
        """

        assert (
            min_center_freq <= max_center_freq
        ), "`min_center_freq` should be no greater than `max_center_freq`"
        assert (
            min_gain_db <= max_gain_db
        ), "`min_gain_db` should be no greater than `max_gain_db`"

        assert 0 < min_q <= 1, "`min_q` should be greater than 0 and less or equal to 1"
        assert 0 < max_q <= 1, "`max_q` should be greater than 0 and less or equal to 1"

        super().__init__(p)

        self.min_center_freq = min_center_freq
        self.max_center_freq = max_center_freq

        self.min_gain_db = min_gain_db
        self.max_gain_db = max_gain_db

        self.min_q = min_q
        self.max_q = max_q

    def _get_biquad_coefficients_from_input_parameters(
        self, center_freq, gain_db, q_factor, sample_rate
    ):
        normalized_frequency = 2 * np.pi * center_freq / sample_rate
        gain = 10 ** (gain_db / 40)
        alpha = np.sin(normalized_frequency) / 2 / q_factor

        b0 = gain * (
            (gain + 1)
            + (gain - 1) * np.cos(normalized_frequency)
            + 2 * np.sqrt(gain) * alpha
        )

        b1 = -2 * gain * ((gain - 1) + (gain + 1) * np.cos(normalized_frequency))

        b2 = gain * (
            (gain + 1)
            + (gain - 1) * np.cos(normalized_frequency)
            - 2 * np.sqrt(gain) * alpha
        )

        a0 = (
            (gain + 1)
            - (gain - 1) * np.cos(normalized_frequency)
            + 2 * np.sqrt(gain) * alpha
        )

        a1 = 2 * ((gain - 1) - (gain + 1) * np.cos(normalized_frequency))

        a2 = (
            (gain + 1)
            - (gain - 1) * np.cos(normalized_frequency)
            - 2 * np.sqrt(gain) * alpha
        )

        # Return it in `sos` format
        sos = np.array([[b0 / a0, b1 / a0, b2 / a0, 1, a1 / a0, a2 / a0]])

        return sos

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)

        center_mel = np.random.uniform(
            low=convert_frequency_to_mel(self.min_center_freq),
            high=convert_frequency_to_mel(self.max_center_freq),
        )
        self.parameters["center_freq"] = convert_mel_to_frequency(center_mel)
        self.parameters["gain_db"] = random.uniform(self.min_gain_db, self.max_gain_db)
        self.parameters["q_factor"] = random.uniform(self.min_q, self.max_q)

    def apply(self, samples, sample_rate):
        assert samples.dtype == np.float32

        sos = self._get_biquad_coefficients_from_input_parameters(
            self.parameters["center_freq"],
            self.parameters["gain_db"],
            self.parameters["q_factor"],
            sample_rate,
        )

        # The processing takes place here
        zi = sosfilt_zi(sos)
        if len(samples.shape) == 1:
            processed_samples, _ = sosfilt(sos, samples, zi=zi * samples[0])
            processed_samples = processed_samples.astype(np.float32)
        else:
            processed_samples = np.zeros_like(samples, dtype=np.float32)
            for chn_idx in range(samples.shape[0]):
                processed_samples[chn_idx, :], _ = sosfilt(
                    sos, samples[chn_idx, :], zi=zi * samples[chn_idx, 0]
                )

        return processed_samples


class FrequencyMask(BaseWaveformTransform):
    """
    Mask some frequency band on the spectrogram.
    Inspired by https://arxiv.org/pdf/1904.08779.pdf

    This transform does basically the same as BandStopFilter
    """

    supports_multichannel = True

    def __init__(self, min_frequency_band=0.0, max_frequency_band=0.5, p=0.5):
        """
        :param min_frequency_band: Minimum bandwidth, float
        :param max_frequency_band: Maximum bandwidth, float
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        warnings.warn(
            "The FrequencyMask class has been deprecated and may be removed in a future"
            " version of audiomentations. You can use BandStopFilter instead. It has"
            " different defaults and different parameter randomization that is better"
            " aligned with human hearing.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.min_frequency_band = min_frequency_band
        self.max_frequency_band = max_frequency_band

    def __butter_bandstop(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], btype="bandstop", output="sos")
        return sos

    def __butter_bandstop_filter(self, data, lowcut, highcut, fs, order=5):
        sos = self.__butter_bandstop(lowcut, highcut, fs, order=order)
        y = sosfilt(sos, data).astype(np.float32)
        return y

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["bandwidth"] = random.randint(
                self.min_frequency_band * sample_rate // 2,
                self.max_frequency_band * sample_rate // 2,
                )
            self.parameters["freq_start"] = random.randint(
                16, sample_rate // 2 - self.parameters["bandwidth"] - 1
            )

    def apply(self, samples, sample_rate):
        bandwidth = self.parameters["bandwidth"]
        freq_start = self.parameters["freq_start"]
        samples = self.__butter_bandstop_filter(
            samples, freq_start, freq_start + bandwidth, sample_rate, order=6
        )
        return samples
