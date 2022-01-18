from .augmentations.spectrogram_transforms import SpecFrequencyMask, SpecChannelShuffle
from .augmentations.transforms import (
    AddBackgroundNoise,
    AddGaussianNoise,
    AddGaussianSNR,
    AddImpulseResponse,
    ApplyImpulseResponse,
    AddShortNoises,
    BandPassFilter,
    BandStopFilter,
    Clip,
    ClippingDistortion,
    FrequencyMask,
    Gain,
    HighPassFilter,
    LoudnessNormalization,
    LowPassFilter,
    Mp3Compression,
    Normalize,
    PitchShift,
    PolarityInversion,
    Resample,
    Reverse,
    Shift,
    TanhDistortion,
    TimeMask,
    TimeStretch,
    Trim,
)
from .core.composition import Compose, SpecCompose, OneOf, SomeOf

__version__ = "0.20.0"
