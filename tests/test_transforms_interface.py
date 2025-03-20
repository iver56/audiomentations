from audiomentations import Normalize


def test_freeze_and_unfreeze_parameters():
    normalizer = Normalize(p=1.0)

    assert normalizer.are_parameters_frozen == False

    normalizer.freeze_parameters()
    assert normalizer.are_parameters_frozen == True

    normalizer.unfreeze_parameters()
    assert normalizer.are_parameters_frozen == False
