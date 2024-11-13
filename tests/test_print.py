import numpy as np
import pytest

from audiomentations import Compose, GainTransition, Shift
from audiomentations.core.utils import format_args


@pytest.mark.parametrize("shape", [(44100,), (1, 22049), (2, 10000)])
def test_print(shape: tuple):
    samples_in = np.random.normal(0.0, 0.5, size=shape).astype(np.float32)
    sample_rate = 44100
    augmenter = Compose([
        GainTransition(),
        Shift()
    ])
    
    augmenter_str = """Compose([\n  GainTransition(p=0.5, min_gain_db=-24.0, max_gain_db=6.0, min_duration=0.2, max_duration=6.0, duration_unit='seconds'),\n  Shift(p=0.5, min_shift=-0.5, max_shift=0.5, shift_unit='fraction', rollover=True, fade_duration=0.005),\n], p=1.0)"""
    
    assert str(augmenter) == augmenter_str
    
    samples_out = augmenter(samples=samples_in, sample_rate=sample_rate)
    assert samples_out.dtype == np.float32
    assert samples_out.shape == shape

