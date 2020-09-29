import unittest

from audiomentations.core.utils import (
    calculate_desired_noise_rms,
    convert_decibels_to_amplitude_ratio,
    get_file_paths,
)
from demo.demo import DEMO_DIR


class TestUtils(unittest.TestCase):
    def test_calculate_desired_noise_rms(self):
        noise_rms = calculate_desired_noise_rms(clean_rms=0.5, snr=6)
        self.assertAlmostEqual(noise_rms, 0.2505936168136362)

    def test_convert_decibels_to_amplitude_ratio(self):
        amplitude_ratio = convert_decibels_to_amplitude_ratio(decibels=-6)
        self.assertAlmostEqual(amplitude_ratio, 0.5011872336272722)

        amplitude_ratio = convert_decibels_to_amplitude_ratio(decibels=6)
        self.assertAlmostEqual(amplitude_ratio, 1.9952623149688795)

    def test_get_file_paths_uppercase_extension(self):
        file_paths = get_file_paths(DEMO_DIR, traverse_subdirectories=False)
        found_it = False
        for file_path in file_paths:
            if file_path.name == "stereo_24bit.WAV":
                found_it = True
                break
        self.assertTrue(found_it)
