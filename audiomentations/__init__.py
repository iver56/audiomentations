from .augmentations.spectrogram_transforms import SpecFrequencyMask, SpecChannelShuffle
from .augmentations.transforms import (
    AddBackgroundNoise,
    AddGaussianNoise,
    AddGaussianSNR,
    AddImpulseResponse,
    ApplyImpulseResponse,
    AddShortNoises,
    Clip,
    ClippingDistortion,
    FrequencyMask,
    Gain,
    InterruptPulse,
    LoudnessNormalization,
    Mp3Compression,
    Normalize,
    PitchShift,
    PolarityInversion,
    Resample,
    Reverse,
    Shift,
    TimeMask,
    TimeStretch,
    Trim,
)
from .core.composition import Compose, SpecCompose

__version__ = "0.17.0"
