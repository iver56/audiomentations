from .augmentations.add_background_noise import AddBackgroundNoise
from .augmentations.add_gaussian_noise import AddGaussianNoise
from .augmentations.add_gaussian_snr import AddGaussianSNR
from .augmentations.add_color_noise import AddColorNoise, NOISE_COLOR_DECAYS
from .augmentations.add_short_noises import AddShortNoises
from .augmentations.adjust_duration import AdjustDuration
from .augmentations.air_absorption import AirAbsorption
from .augmentations.aliasing import Aliasing
from .augmentations.apply_impulse_response import ApplyImpulseResponse
from .augmentations.band_pass_filter import BandPassFilter
from .augmentations.band_stop_filter import BandStopFilter
from .augmentations.bit_crush import BitCrush
from .augmentations.clip import Clip
from .augmentations.clipping_distortion import ClippingDistortion
from .augmentations.gain import Gain
from .augmentations.gain_transition import GainTransition
from .augmentations.high_pass_filter import HighPassFilter
from .augmentations.high_shelf_filter import HighShelfFilter
from .augmentations.lambda_transform import Lambda
from .augmentations.limiter import Limiter
from .augmentations.loudness_normalization import LoudnessNormalization
from .augmentations.low_pass_filter import LowPassFilter
from .augmentations.low_shelf_filter import LowShelfFilter
from .augmentations.mp3_compression import Mp3Compression
from .augmentations.normalize import Normalize
from .augmentations.padding import Padding
from .augmentations.peaking_filter import PeakingFilter
from .augmentations.pitch_shift import PitchShift
from .augmentations.polarity_inversion import PolarityInversion
from .augmentations.repeat_part import RepeatPart
from .augmentations.resample import Resample
from .augmentations.reverse import Reverse
from .augmentations.room_simulator import RoomSimulator
from .augmentations.seven_band_parametric_eq import SevenBandParametricEQ
from .augmentations.shift import Shift
from .augmentations.tanh_distortion import TanhDistortion
from .augmentations.time_mask import TimeMask
from .augmentations.time_stretch import TimeStretch
from .augmentations.trim import Trim
from .core.composition import Compose, SpecCompose, OneOf, SomeOf
from .spec_augmentations.spec_channel_shuffle import SpecChannelShuffle
from .spec_augmentations.spec_frequency_mask import SpecFrequencyMask

__version__ = "0.39.0"
