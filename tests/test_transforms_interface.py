import unittest

from audiomentations.augmentations.transforms import Normalize
from audiomentations.core.composition import Compose


class TestTransformsInterface(unittest.TestCase):
    def test_assigned_id(self):
        augmenter = Compose([Normalize(p=1.0)])
        self.assertIn("Normalize_", augmenter.transforms[0].id)
