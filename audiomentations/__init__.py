from .augmentations.spectrogram_transforms import SpecFrequencyMask, SpecChannelShuffle
from .augmentations.transforms import (
    AddBackgroundNoise,
    AddGaussianNoise,
    AddGaussianSNR,
    AddImpulseResponse,
    AddShortNoises,
    ClippingDistortion,
    FrequencyMask,
    Gain,
    LoudnessNormalization,
    Mp3Compression,
    Normalize,
    PitchShift,
    PolarityInversion,
    Resample,
    Shift,
    TimeMask,
    TimeStretch,
    Trim,
)
from .core.composition import Compose

__version__ = "0.14.0"
