from .core.composition import Compose
from .augmentations.transforms import (
    AddImpulseResponse,
    FrequencyMask,
    TimeMask,
    AddGaussianSNR,
    AddGaussianNoise,
    TimeStretch,
    PitchShift,
    Shift,
    Normalize,
    Trim,
    Resample,
    ClippingDistortion,
    AddBackgroundNoise,
    AddShortNoises,
    PolarityInversion,
    Gain,
)

__version__ = "0.11.0"
