import unittest

from audiomentations.augmentations.transforms import Normalize


class TestTransformsInterface(unittest.TestCase):
    def test_freeze_and_unfreeze_parameters(self):
        normalizer = Normalize(p=1.0)

        self.assertFalse(normalizer.are_parameters_frozen)

        normalizer.freeze_parameters()
        self.assertTrue(normalizer.are_parameters_frozen)

        normalizer.unfreeze_parameters()
        self.assertFalse(normalizer.are_parameters_frozen)
