from .augmentations.filters import (
    PeakingFilter,
    LowPassFilter,
    HighPassFilter,
    LowShelfFilter,
    HighShelfFilter,
    FrequencyMask,
    BandPassFilter,
    BandStopFilter,
)
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
    Gain,
    LoudnessNormalization,
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

__version__ = "0.21.0"
