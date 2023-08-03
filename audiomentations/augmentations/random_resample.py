import random
import librosa
import numpy as np
from audiomentations.core.transforms_interface import BaseWaveformTransform


class RandomResample(BaseWaveformTransform):
    """
    Resample signal using librosa.core.resample in one direction than resample back to original shape.
    """

    supports_multichannel = True

    def __init__(self, min_sample_rate=8000, max_sample_rate=44100, methods=('kaiser_fast', 'fft'), p=0.5):
        """
        :param min_sample_rate: int, Minimum sample rate
        :param max_sample_rate: int, Maximum sample rate
        :param methods: int, Resample type. If several added it will be chosen randonly.
        All possible values see in: librosa.core.audio.resample
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        assert min_sample_rate <= max_sample_rate
        self.min_sample_rate = min_sample_rate
        self.max_sample_rate = max_sample_rate
        self.methods = methods

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            self.parameters["target_sample_rate"] = random.randint(
                self.min_sample_rate, self.max_sample_rate
            )
            self.parameters["target_method"] = random.choice(
                self.methods
            )

    def apply(self, samples, sample_rate):
        orig_shape = samples.shape
        samples = librosa.core.resample(
            samples,
            orig_sr=sample_rate,
            target_sr=self.parameters["target_sample_rate"],
            res_type=self.parameters["target_method"],
        )
        samples = librosa.core.resample(
            samples,
            orig_sr=self.parameters["target_sample_rate"],
            target_sr=sample_rate,
            res_type=self.parameters["target_method"],
        )
        new_shape = samples.shape
        # print(self.parameters["target_sample_rate"], self.parameters["target_method"])
        if orig_shape[-1] < new_shape[-1]:
            # print('Shape less! {} != {}'.format(orig_shape, new_shape))
            samples = samples[..., :orig_shape[-1]]
        elif orig_shape[-1] > new_shape[-1]:
            # print('Shape more! {} != {}'.format(orig_shape, new_shape))
            samples = np.pad(samples, ((0, 0), (0, orig_shape[-1] - new_shape[-1])), 'constant')
        return samples


"""
Time statistics for different methods:
Aug: RandomResample Type: kaiser_best Time: 56.39 sec Per sample: 0.563885 sec
Aug: RandomResample Type: kaiser_fast Time: 15.02 sec Per sample: 0.150230 sec
Aug: RandomResample Type: scipy/fft Time: 12.38 sec Per sample: 0.123800 sec
Aug: RandomResample Type: linear Time: 2.48 sec Per sample: 0.024790 sec
Aug: RandomResample Type: zero_order_hold Time: 1.83 sec Per sample: 0.018273 sec
Aug: RandomResample Type: sinc_best Time: 70.59 sec Per sample: 0.705902 sec
Aug: RandomResample Type: sinc_medium Time: 19.38 sec Per sample: 0.193758 sec
Aug: RandomResample Type: sinc_fastest Time: 10.09 sec Per sample: 0.100885 sec
Aug: RandomResample Type: soxr_vhq Time: 5.58 sec Per sample: 0.055791 sec
Aug: RandomResample Type: soxr_hq Time: 2.59 sec Per sample: 0.025902 sec
Aug: RandomResample Type: soxr_mq Time: 2.42 sec Per sample: 0.024174 sec
Aug: RandomResample Type: soxr_lq Time: 2.36 sec Per sample: 0.023578 sec
Aug: RandomResample Type: soxr_qq Time: 1.82 sec Per sample: 0.018217 sec
"""
