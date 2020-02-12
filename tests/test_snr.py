import unittest

from audiomentations.core.snr import SNR


class TestSNR(unittest.TestCase):
    def test_SNR_unit_conversion(self):
        # Initialize without args
        with self.assertRaises(Exception):
            SNR()

        # Initialize without too many args
        with self.assertRaises(Exception):
            SNR(decibels=-6, amplitude_ratio=0.234)

        # Initialize with amplitude_ratio
        snr = SNR(amplitude_ratio=0.25)
        self.assertAlmostEqual(snr.amplitude_ratio, 0.25)
        self.assertAlmostEqual(snr.decibels, -12.041199826559248)

        # Initialize with decibels
        snr = SNR(decibels=-6.0)
        self.assertAlmostEqual(snr.decibels, -6.0)
        self.assertAlmostEqual(snr.amplitude_ratio, 0.5011872336272722)

        # Set amplitude_ratio after init
        snr.amplitude_ratio = 0.5
        self.assertAlmostEqual(snr.amplitude_ratio, 0.5)
        self.assertAlmostEqual(snr.decibels, -6.020599913279624)

        # Set decibels after init
        snr.decibels = 24
        self.assertAlmostEqual(snr.decibels, 24.0)
        self.assertAlmostEqual(snr.amplitude_ratio, 15.848931924611133)
