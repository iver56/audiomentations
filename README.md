# Audiomentations

[![Build status](https://img.shields.io/circleci/project/github/iver56/audiomentations/master.svg)](https://circleci.com/gh/iver56/audiomentations) [![Code coverage](https://img.shields.io/codecov/c/github/iver56/audiomentations/master.svg)](https://codecov.io/gh/iver56/audiomentations) [![Code Style: Black](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/ambv/black) [![Licence: MIT](https://img.shields.io/pypi/l/audiomentations)](https://github.com/iver56/audiomentations/blob/master/LICENSE)

A Python library for audio data augmentation. Inspired by [albumentations](https://github.com/albu/albumentations). Useful for machine learning.

# Setup

![Python version support](https://img.shields.io/pypi/pyversions/audiomentations)
[![PyPI version](https://img.shields.io/pypi/v/audiomentations.svg?style=flat)](https://pypi.org/project/audiomentations/)
[![Number of downloads from PyPI per month](https://img.shields.io/pypi/dm/audiomentations.svg?style=flat)](https://pypi.org/project/audiomentations/)

`pip install audiomentations`

# Usage example

```python
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import numpy as np

SAMPLE_RATE = 16000

augmenter = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
])

samples = np.zeros((20,), dtype=np.float32)
samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
```

Go to [audiomentations/augmentations/transforms.py](https://github.com/iver56/audiomentations/blob/master/audiomentations/augmentations/transforms.py) to see which transforms you can apply.

# Version history

## v0.11.0 (2020-08-27)

* Implement `Gain` and `PolarityInversion`. Thanks to Spijkervet for the inspiration.

## v0.10.1 (2020-07-27)

* Improve the performance of `AddBackgroundNoise` and `AddShortNoises` by optimizing the implementation of `calculate_rms`.
* Improve compatibility of output files written by the demo script. Thanks to xwJohn.
* Fix division by zero bug in `Normalize`. Thanks to ZFTurbo.

## v0.10.0 (2020-05-05)

* Breaking change: `AddImpulseResponse`, `AddBackgroundNoise` and `AddShortNoises` now include subfolders when searching for files. This is useful when your sound files are organized in subfolders.
* `AddImpulseResponse`, `AddBackgroundNoise` and `AddShortNoises` now support aiff files in addition to flac, mp3, ogg and wav
* Fix filter instability bug in `FrequencyMask`. Thanks to kvilouras.

## v0.9.0 (2020-02-20)

* Disregard non-audio files when looking for impulse response files
* Remember randomized/chosen effect parameters. This allows for freezing the parameters and applying the same effect to multiple sounds. Use transform.freeze_parameters() and transform.unfreeze_parameters() for this.
* Fix a bug in `ClippingDistortion` where the min_percentile_threshold was not respected as expected.
* Implement transform.serialize_parameters(). Useful for when you want to store metadata on how a sound was perturbed.
* Switch to a faster convolve implementation. This makes `AddImpulseResponse` significantly faster.
* Add a rollover parameter to `Shift`. This allows for introducing silence instead of a wrapped part of the sound.
* Expand supported range of librosa versions
* Add support for flac in `AddImpulseResponse`
* Implement `AddBackgroundNoise` transform. Useful for when you want to add background noise to all of your sound. You need to give it a folder of background noises to choose from.
* Implement `AddShortNoises`. Useful for when you want to add (bursts of) short noise sounds to your input audio.
* Improve handling of empty input

## v0.8.0 (2020-01-28)

* Add shuffle parameter in `Composer`
* Add `Resample` transformation
* Add `ClippingDistortion` transformation
* Add `SmoothFadeTimeMask` as alternative to `TimeMask`

Thanks to askskro

## v0.7.0 (2020-01-14)

Add new transforms:

* `AddImpulseResponse`
* `FrequencyMask`
* `TimeMask`
* `AddGaussianSNR`

Thanks to karpnv

## v0.6.0 (2019-05-27)

* Implement peak normalization

## v0.5.0 (2019-02-23)

* Implement `Shift` transform
* Ensure p is within bounds

## v0.4.0 (2019-02-19)

* Implement `PitchShift` transform
* Fix output dtype of `AddGaussianNoise`

## v0.3.0 (2019-02-19)

Implement `leave_length_unchanged` in `TimeStretch`

## v0.2.0 (2019-02-18)

* Add `TimeStretch` transform
* Parametrize `AddGaussianNoise`

## v0.1.0 (2019-02-15)

Initial release. Includes only one transform: `AddGaussianNoise`


# Development

Install the dependencies specified in `requirements.txt`

## Code style

Format the code with `black`

## Run tests and measure code coverage

`pytest`

## Generate demo sounds for empirical evaluation

`python -m demo.demo`

# Alternatives

* [spec_augment](https://github.com/zcaceres/spec_augment)
* [muda](https://github.com/bmcfee/muda)
* [nlpaug](https://github.com/makcedward/nlpaug)
* [python-audio-effects](https://github.com/carlthome/python-audio-effects)
* [audio_degrader](https://github.com/emilio-molina/audio_degrader)
* [WavAugment](https://github.com/facebookresearch/WavAugment)
