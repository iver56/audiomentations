# Audiomentations

[![Build status](https://img.shields.io/circleci/project/github/iver56/audiomentations/main.svg)](https://circleci.com/gh/iver56/audiomentations)
[![Code coverage](https://img.shields.io/codecov/c/github/iver56/audiomentations/main.svg)](https://codecov.io/gh/iver56/audiomentations)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/psf/black)
[![Licence: MIT](https://img.shields.io/pypi/l/audiomentations)](https://github.com/iver56/audiomentations/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15056865.svg)](https://doi.org/10.5281/zenodo.15056865)

Audiomentations is a Python library for audio data augmentation, built to be fast and easy to use - its API is inspired by
[albumentations](https://github.com/albu/albumentations). It's useful for making audio deep learning models work well in the real world, not just in the lab.
Audiomentations runs on CPU, supports mono audio and multichannel audio and integrates well in training pipelines,
such as those built with TensorFlow/Keras or PyTorch. It has helped users achieve
world-class results in Kaggle competitions and is trusted by companies building next-generation audio products with AI.

Need a Pytorch-specific alternative with GPU support? Check out [torch-audiomentations](https://github.com/asteroid-team/torch-audiomentations)!

# Setup

![Python version support](https://img.shields.io/pypi/pyversions/audiomentations)
[![PyPI version](https://img.shields.io/pypi/v/audiomentations.svg?style=flat)](https://pypi.org/project/audiomentations/)
[![Number of downloads from PyPI per month](https://img.shields.io/pypi/dm/audiomentations.svg?style=flat)](https://pypi.org/project/audiomentations/)
![os: Linux, macOS, Windows](https://img.shields.io/badge/OS-Linux%20%28arm%20%26%20x86%29%20|%20macOS%20%28arm%29%20|%20Windows%20%28x86%29-blue)

`pip install audiomentations`

# Usage example

```python
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import numpy as np

augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(p=0.5),
])

# Generate 2 seconds of dummy audio for the sake of example
samples = np.random.uniform(low=-0.2, high=0.2, size=(32000,)).astype(np.float32)

# Augment/transform/perturb the audio data
augmented_samples = augment(samples=samples, sample_rate=16000)
```

# Documentation

The API documentation, along with guides, example code, illustrations and example sounds, is available at [https://iver56.github.io/audiomentations/](https://iver56.github.io/audiomentations/)

# Transforms

* [AddBackgroundNoise](https://iver56.github.io/audiomentations/waveform_transforms/add_background_noise/): Mixes in another sound to add background noise
* [AddColorNoise](https://iver56.github.io/audiomentations/waveform_transforms/add_color_noise/): Adds noise with specific color
* [AddGaussianNoise](https://iver56.github.io/audiomentations/waveform_transforms/add_gaussian_noise/): Adds gaussian noise to the audio samples
* [AddGaussianSNR](https://iver56.github.io/audiomentations/waveform_transforms/add_gaussian_snr/): Injects gaussian noise using a randomly chosen signal-to-noise ratio
* [AddShortNoises](https://iver56.github.io/audiomentations/waveform_transforms/add_short_noises/): Mixes in various short noise sounds
* [AdjustDuration](https://iver56.github.io/audiomentations/waveform_transforms/adjust_duration/): Trims or pads the audio to fit a target duration
* [AirAbsorption](https://iver56.github.io/audiomentations/waveform_transforms/air_absorption/): Applies frequency-dependent attenuation simulating air absorption
* [Aliasing](https://iver56.github.io/audiomentations/waveform_transforms/aliasing/): Produces aliasing artifacts by downsampling without low-pass filtering and then upsampling
* [ApplyImpulseResponse](https://iver56.github.io/audiomentations/waveform_transforms/apply_impulse_response/): Convolves the audio with a randomly chosen impulse response
* [BandPassFilter](https://iver56.github.io/audiomentations/waveform_transforms/band_pass_filter/): Applies band-pass filtering within randomized parameters
* [BandStopFilter](https://iver56.github.io/audiomentations/waveform_transforms/band_stop_filter/): Applies band-stop (notch) filtering within randomized parameters
* [BitCrush](https://iver56.github.io/audiomentations/waveform_transforms/bit_crush/): Applies bit reduction without dithering
* [Clip](https://iver56.github.io/audiomentations/waveform_transforms/clip/): Clips audio samples to specified minimum and maximum values
* [ClippingDistortion](https://iver56.github.io/audiomentations/waveform_transforms/clipping_distortion/): Distorts the signal by clipping a random percentage of samples
* [Gain](https://iver56.github.io/audiomentations/waveform_transforms/gain/): Multiplies the audio by a random gain factor
* [GainTransition](https://iver56.github.io/audiomentations/waveform_transforms/gain_transition/): Gradually changes the gain over a random time span
* [HighPassFilter](https://iver56.github.io/audiomentations/waveform_transforms/high_pass_filter/): Applies high-pass filtering within randomized parameters
* [HighShelfFilter](https://iver56.github.io/audiomentations/waveform_transforms/high_shelf_filter/): Applies a high shelf filter with randomized parameters
* [Lambda](https://iver56.github.io/audiomentations/waveform_transforms/lambda/): Applies a user-defined transform
* [Limiter](https://iver56.github.io/audiomentations/waveform_transforms/limiter/): Applies dynamic range compression limiting the audio signal
* [LoudnessNormalization](https://iver56.github.io/audiomentations/waveform_transforms/loudness_normalization/): Applies gain to match a target loudness
* [LowPassFilter](https://iver56.github.io/audiomentations/waveform_transforms/low_pass_filter/): Applies low-pass filtering within randomized parameters
* [LowShelfFilter](https://iver56.github.io/audiomentations/waveform_transforms/low_shelf_filter/): Applies a low shelf filter with randomized parameters
* [Mp3Compression](https://iver56.github.io/audiomentations/waveform_transforms/mp3_compression/): Compresses the audio to lower the quality
* [Normalize](https://iver56.github.io/audiomentations/waveform_transforms/normalize/): Applies gain so that the highest signal level becomes 0 dBFS
* [Padding](https://iver56.github.io/audiomentations/waveform_transforms/padding/): Replaces a random part of the beginning or end with padding
* [PeakingFilter](https://iver56.github.io/audiomentations/waveform_transforms/peaking_filter/): Applies a peaking filter with randomized parameters
* [PitchShift](https://iver56.github.io/audiomentations/waveform_transforms/pitch_shift/): Shifts the pitch up or down without changing the tempo
* [PolarityInversion](https://iver56.github.io/audiomentations/waveform_transforms/polarity_inversion/): Flips the audio samples upside down, reversing their polarity
* [RepeatPart](https://iver56.github.io/audiomentations/waveform_transforms/repeat_part/): Repeats a subsection of the audio a number of times
* [Resample](https://iver56.github.io/audiomentations/waveform_transforms/resample/): Resamples the signal to a randomly chosen sampling rate
* [Reverse](https://iver56.github.io/audiomentations/waveform_transforms/reverse/): Reverses the audio along its time axis
* [RoomSimulator](https://iver56.github.io/audiomentations/waveform_transforms/room_simulator/): Simulates the effect of a room on an audio source
* [SevenBandParametricEQ](https://iver56.github.io/audiomentations/waveform_transforms/seven_band_parametric_eq/): Adjusts the volume of 7 frequency bands
* [Shift](https://iver56.github.io/audiomentations/waveform_transforms/shift/): Shifts the samples forwards or backwards
* [TanhDistortion](https://iver56.github.io/audiomentations/waveform_transforms/tanh_distortion/): Applies tanh distortion to distort the signal
* [TimeMask](https://iver56.github.io/audiomentations/waveform_transforms/time_mask/): Makes a random part of the audio silent
* [TimeStretch](https://iver56.github.io/audiomentations/waveform_transforms/time_stretch/): Changes the speed without changing the pitch
* [Trim](https://iver56.github.io/audiomentations/waveform_transforms/trim/): Trims leading and trailing silence from the audio

# Changelog

## [0.41.0] - 2025-05-05

### Added

* Add support for NumPy 2.x
* Add `weights` parameter to `OneOf`. This lets you guide the probability of each transform being chosen.

### Changed

* Improve type hints

#### :warning: The `TimeMask` transform has been changed significantly:

* **Breaking change**: Remove `fade` parameter. `fade_duration=0.0` now denotes disabled fading.
* Enable fading by default
* Apply a smooth fade curve instead of a linear one
* Add `mask_location` parameter
* Change the default value of `min_band_part` from 0.0 to 0.01
* Change the default value of `max_band_part` from 0.5 to 0.2
* ~50% faster

The following examples show how you can adapt your code when upgrading from <=v0.40.0 to >=v0.41.0:

| <= 0.40.0                                                    | >= 0.41.0                                                             |
|--------------------------------------------------------------|-----------------------------------------------------------------------|
| `TimeMask(min_band_part=0.1, max_band_part=0.15, fade=True)` | `TimeMask(min_band_part=0.1, max_band_part=0.15, fade_duration=0.01)` |
| `TimeMask()`                                                 | `TimeMask(min_band_part=0.0, max_band_part=0.5, fade_duration=0.0)`   |

### Removed

* `SpecCompose`, `SpecChannelShuffle` and `SpecFrequencyMask` have been removed. You can read more about this here: [#391](https://github.com/iver56/audiomentations/pull/391)

For the full changelog, including older versions, see [https://iver56.github.io/audiomentations/changelog/](https://iver56.github.io/audiomentations/changelog/)

# Acknowledgements

Thanks to [Nomono](https://nomono.co/) for backing audiomentations.

Thanks to [all contributors](https://github.com/iver56/audiomentations/graphs/contributors) who help improving audiomentations.
