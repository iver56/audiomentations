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
    Mp3Compression,
)
from .core.composition import Compose

__version__ = "0.12.1"
