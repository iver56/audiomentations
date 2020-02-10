import unittest

from audiomentations import calculate_desired_noise_rms


class TestUtils(unittest.TestCase):
    def test_calculate_desired_noise_rms(self):
        noise_rms = calculate_desired_noise_rms(clean_rms=0.5, snr=6)
        self.assertAlmostEqual(noise_rms, 0.2505936168136362)
