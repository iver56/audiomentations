import random

import librosa

from audiomentations.core.transforms_interface import BaseWaveformTransform


class BandLimitWithTwoPhaseResample(BaseWaveformTransform):
    RESAMPLE_TYPES = ["soxr_vhq", "soxr_hq", "soxr_mq", "soxr_lq", "soxr_qq",
                         "kaiser_best", "kaiser_fast", "fft", "polyphase", "linear", "zero_order_hold",
                         "sinc_best", "sinc_medium", "sinc_fastest"]

    """
    Band limit with two phase resample.
    Phase 1: Downsample to a random sample rate between min_sample_rate and max_sample_rate
    Phase 2: Upsample back to the original sample rate

    If the random sampling rate between min_sample_rate and max_sample_rate is greater than the original sample rate, 
    the audio will be upsamled first and then downsampled.
    """

    supports_multichannel = True

    def __init__(self, 
                 min_sample_rate=8000, 
                 max_sample_rate=44100,
                 res_types=RESAMPLE_TYPES,
                 p=0.5):
        """
        :param min_sample_rate: int, Minimum sample rate
        :param max_sample_rate: int, Maximum sample rate
        :param res_types: [None, "all" or list of resample types], Resample types to use. 
            Should be from librosa resample res_types
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        assert min_sample_rate <= max_sample_rate
        self.min_sample_rate = min_sample_rate
        self.max_sample_rate = max_sample_rate
        
        self.res_types = res_types
        if self.res_types == 'all':
            self.res_types = self.RESAMPLE_TYPES

        if self.res_types:
            for i in self.res_types:
                assert i in self.RESAMPLE_TYPES

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["target_sample_rate"] = random.randint(
                self.min_sample_rate, self.max_sample_rate
            )
            
            if self.res_types:
                self.parameters["res_type_down"] = random.choice(self.res_types)
                self.parameters["res_type_up"] = random.choice(self.res_types)
            else:
                self.parameters["res_type_down"] = None
                self.parameters["res_type_up"] = None

    def apply(self, samples, sample_rate):
        downsampled_samples = librosa.core.resample(
            samples,
            orig_sr=sample_rate,
            target_sr=self.parameters["target_sample_rate"],
            res_type=self.parameters["res_type_down"],
        )

        restored_samples = librosa.core.resample(
            downsampled_samples,
            orig_sr=self.parameters["target_sample_rate"],
            target_sr=sample_rate,
            res_type=self.parameters["res_type_up"],
        )

        if samples.shape != restored_samples.shape:
            restored_samples = librosa.util.fix_length(restored_samples, size=samples.shape[-1])
        assert samples.shape == restored_samples.shape
        return restored_samples