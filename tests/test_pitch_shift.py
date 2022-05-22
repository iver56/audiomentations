from copy import deepcopy

import numpy as np
from numpy.testing import assert_array_equal

from audiomentations import PitchShift, Compose


class TestPitchShift:
    def test_apply_pitch_shift(self):
        samples = np.zeros((2048,), dtype=np.float32)
        sample_rate = 16000
        augmenter = Compose([PitchShift(min_semitones=-2, max_semitones=-1, p=1.0)])
        samples = augmenter(samples=samples, sample_rate=sample_rate)

        assert samples.dtype == np.float32
        assert samples.shape[-1] == 2048

    def test_apply_pitch_shift_multichannel(self):
        num_channels = 3
        samples = np.random.normal(0, 0.1, size=(num_channels, 5555)).astype(np.float32)
        sample_rate = 16000
        augmenter = Compose([PitchShift(min_semitones=1, max_semitones=2, p=1.0)])
        samples_out = augmenter(samples=samples, sample_rate=sample_rate)

        assert samples_out.dtype == np.float32
        assert samples_out.shape == samples.shape
        for i in range(num_channels):
            assert not np.allclose(samples[i], samples_out[i])

    def test_freeze_parameters(self):
        """
        Test that the transform can freeze its parameters, e.g. to apply the effect with the
        same parameters to multiple sounds.
        """
        samples = np.sin(np.linspace(0, 440 * 2 * np.pi, 8000)).astype(np.float32)
        sample_rate = 16000
        augmenter = Compose([PitchShift(min_semitones=1, max_semitones=12, p=1.0)])

        first_samples = augmenter(samples=samples, sample_rate=sample_rate)
        first_parameters = deepcopy(augmenter.transforms[0].parameters)

        augmenter.transforms[0].min_semitones = -12
        augmenter.transforms[0].max_semitones = -1
        augmenter.transforms[0].are_parameters_frozen = True
        second_samples = augmenter(samples=samples, sample_rate=sample_rate)

        assert first_parameters == augmenter.transforms[0].parameters
        assert_array_equal(first_samples, second_samples)
