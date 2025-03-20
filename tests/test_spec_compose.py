import numpy as np
from numpy.testing import assert_array_equal

from audiomentations import SpecChannelShuffle, SpecFrequencyMask, SpecCompose


def test_freeze_and_unfreeze_parameters():
    spectrogram = np.random.random((256, 256, 3))
    augmenter = SpecCompose(
        [
            SpecChannelShuffle(p=1.0),
            SpecFrequencyMask(p=1.0),
        ]
    )
    perturbed_samples1 = augmenter(magnitude_spectrogram=spectrogram)
    augmenter.freeze_parameters()
    for transform in augmenter.transforms:
        assert transform.are_parameters_frozen == True
    perturbed_samples2 = augmenter(magnitude_spectrogram=spectrogram)

    assert_array_equal(perturbed_samples1, perturbed_samples2)

    augmenter.unfreeze_parameters()
    for transform in augmenter.transforms:
        assert transform.are_parameters_frozen == False

def test_randomize_parameters_and_apply():
    spectrogram = np.random.random((256, 256, 3))
    augmenter = SpecCompose(
        [
            SpecChannelShuffle(p=1.0),
            SpecFrequencyMask(p=1.0),
        ]
    )
    augmenter.freeze_parameters()
    augmenter.randomize_parameters(magnitude_spectrogram=spectrogram)

    parameters = [transform.parameters for transform in augmenter.transforms]

    perturbed_samples1 = augmenter(magnitude_spectrogram=spectrogram)
    perturbed_samples2 = augmenter(magnitude_spectrogram=spectrogram)

    assert_array_equal(perturbed_samples1, perturbed_samples2)

    augmenter.unfreeze_parameters()

    for transform_parameters, transform in zip(parameters, augmenter.transforms):
        assert transform_parameters == transform.parameters
        assert transform.are_parameters_frozen == False
