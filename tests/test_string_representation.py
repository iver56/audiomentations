from audiomentations import Compose, GainTransition, Shift


def test_print_compose():
    augmenter = Compose([GainTransition(), Shift()])
    assert (
        str(augmenter)
        == """Compose([
  GainTransition(p=0.5, min_gain_db=-24.0, max_gain_db=6.0, min_duration=0.2, max_duration=6.0, duration_unit='seconds'),
  Shift(p=0.5, min_shift=-0.5, max_shift=0.5, shift_unit='fraction', rollover=True, fade_duration=0.005),
], p=1.0)"""
    )
