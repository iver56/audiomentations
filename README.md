# Audiomentations

[![Build status](https://img.shields.io/circleci/project/github/iver56/audiomentations/master.svg)](https://circleci.com/gh/iver56/audiomentations) [![Code coverage](https://img.shields.io/codecov/c/github/iver56/audiomentations/master.svg)](https://codecov.io/gh/iver56/audiomentations) [![Code Style: Black](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/ambv/black) [![Licence: MIT](https://img.shields.io/pypi/l/audiomentations)](https://github.com/iver56/audiomentations/blob/master/LICENSE)

A Python library for audio data augmentation. Inspired by
[albumentations](https://github.com/albu/albumentations). Useful for deep learning. Runs on
CPU. Supports mono audio and [partially multichannel audio](#multichannel-audio). Can be
integrated in training pipelines in e.g. Tensorflow/Keras or Pytorch. Has helped people get
world-class results in Kaggle competitions. Is used by companies making next-generation audio
products.

Need a Pytorch alternative with GPU support? Check out [torch-audiomentations](https://github.com/asteroid-team/torch-audiomentations)!

# Setup

![Python version support](https://img.shields.io/pypi/pyversions/audiomentations)
[![PyPI version](https://img.shields.io/pypi/v/audiomentations.svg?style=flat)](https://pypi.org/project/audiomentations/)
[![Number of downloads from PyPI per month](https://img.shields.io/pypi/dm/audiomentations.svg?style=flat)](https://pypi.org/project/audiomentations/)

`pip install audiomentations`

## Optional requirements

Some features have extra dependencies. Extra python package dependencies can be installed by running

`pip install audiomentations[extras]`

| Feature | Extra dependencies |
| ------- | ---------------- |
| Load 24-bit wav files fast | `wavio` |
| `LoudnessNormalization` | `pyloudnorm` |
| `Mp3Compression` | `ffmpeg` and [`pydub` or `lameenc`] |

Note: `ffmpeg` can be installed via e.g. conda or from [the official ffmpeg download page](http://ffmpeg.org/download.html).

# Usage example

```python
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import numpy as np

SAMPLE_RATE = 16000

augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
])

# Generate 2 seconds of dummy audio for the sake of example
samples = np.random.uniform(low=-0.2, high=0.2, size=(32000,)).astype(np.float32)

# Augment/transform/perturb the audio data
augmented_samples = augment(samples=samples, sample_rate=SAMPLE_RATE)
```

Go to [audiomentations/augmentations/transforms.py](https://github.com/iver56/audiomentations/blob/master/audiomentations/augmentations/transforms.py) to see the waveform transforms you can apply, and what arguments they have.

See [audiomentations/augmentations/spectrogram_transforms.py](https://github.com/iver56/audiomentations/blob/master/audiomentations/augmentations/spectrogram_transforms.py) for spectrogram transforms. 

# Waveform transforms

## `AddBackgroundNoise`

_Added in v0.9.0_

Mix in another sound, e.g. a background noise. Useful if your original sound is clean and
you want to simulate an environment where background noise is present.

Can also be used for mixup, as in https://arxiv.org/pdf/1710.09412.pdf

A folder of (background noise) sounds to be mixed in must be specified. These sounds should
ideally be at least as long as the input sounds to be transformed. Otherwise, the background
sound will be repeated, which may sound unnatural.

Note that the gain of the added noise is relative to the amount of signal in the input. This
implies that if the input is completely silent, no noise will be added.

## `AddGaussianNoise`

_Added in v0.1.0_

Add gaussian noise to the samples

## `AddGaussianSNR`

_Added in v0.7.0_

Add gaussian noise to the samples with random Signal to Noise Ratio (SNR)

## `AddImpulseResponse`

_Added in v0.7.0_ 

Convolve the audio with a random impulse response.
Impulse responses can be created using e.g. http://tulrich.com/recording/ir_capture/
Some datasets of impulse responses are publicly available :
- [EchoThief](http://www.echothief.com/) containing 115 impulse responses acquired in a wide range of locations.
- [The MIT McDermott](https://mcdermottlab.mit.edu/Reverb/IR_Survey.html) dataset containing 271 impulse responses acquired in everyday places.

Impulse responses are represented as wav files in the given ir_path.

## `AddShortNoises`

_Added in v0.9.0_

Mix in various (bursts of overlapping) sounds with random pauses between. Useful if your
original sound is clean and you want to simulate an environment where short noises sometimes
occur.

A folder of (noise) sounds to be mixed in must be specified.

## `ClippingDistortion`

_Added in v0.8.0_

Distort signal by clipping a random percentage of points

The percentage of points that will ble clipped is drawn from a uniform distribution between
the two input parameters min_percentile_threshold and max_percentile_threshold. If for instance
30% is drawn, the samples are clipped if they're below the 15th or above the 85th percentile.

## `FrequencyMask`

_Added in v0.7.0_

Mask some frequency band on the spectrogram.
Inspired by https://arxiv.org/pdf/1904.08779.pdf

## `Gain`
_Added in v0.11.0_

Multiply the audio by a random amplitude factor to reduce or increase the volume. This
technique can help a model become somewhat invariant to the overall gain of the input audio.

Warning: This transform can return samples outside the [-1, 1] range, which may lead to
clipping or wrap distortion, depending on what you do with the audio in a later stage.
See also https://en.wikipedia.org/wiki/Clipping_(audio)#Digital_clipping

## `Mp3Compression`

_Added in v0.12.0_

Compress the audio using an MP3 encoder to lower the audio quality. This may help machine
learning models deal with compressed, low-quality audio.

This transform depends on either lameenc or pydub/ffmpeg.

Note that bitrates below 32 kbps are only supported for low sample rates (up to 24000 hz).

Note: When using the lameenc backend, the output may be slightly longer than the input due
to the fact that the LAME encoder inserts some silence at the beginning of the audio.

## `LoudnessNormalization`

_Added in v0.14.0_

Apply a constant amount of gain to match a specific loudness. This is an implementation of
ITU-R BS.1770-4.

Warning: This transform can return samples outside the [-1, 1] range, which may lead to
clipping or wrap distortion, depending on what you do with the audio in a later stage.
See also https://en.wikipedia.org/wiki/Clipping_(audio)#Digital_clipping

## `Normalize`

_Added in v0.6.0_

Apply a constant amount of gain, so that highest signal level present in the sound becomes
0 dBFS, i.e. the loudest level allowed if all samples must be between -1 and 1. Also known
as peak normalization.

## `PitchShift`

_Added in v0.4.0_

Pitch shift the sound up or down without changing the tempo

## `PolarityInversion`

_Added in v0.11.0_

Flip the audio samples upside-down, reversing their polarity. In other words, multiply the
waveform by -1, so negative values become positive, and vice versa. The result will sound
the same compared to the original when played back in isolation. However, when mixed with
other audio sources, the result may be different. This waveform inversion technique
is sometimes used for audio cancellation or obtaining the difference between two waveforms.
However, in the context of audio data augmentation, this transform can be useful when
training phase-aware machine learning models.

## `Resample`

_Added in v0.8.0_

Resample signal using librosa.core.resample

To do downsampling only set both minimum and maximum sampling rate lower than original
sampling rate and vice versa to do upsampling only.

## `Shift`

_Added in v0.5.0_

Shift the samples forwards or backwards, with or without rollover

## `TimeMask`

_Added in v0.7.0_

Make a randomly chosen part of the audio silent.
Inspired by https://arxiv.org/pdf/1904.08779.pdf

## `TimeStretch`

_Added in v0.2.0_

Time stretch the signal without changing the pitch

## `Trim`

_Added in v0.7.0_

Trim leading and trailing silence from an audio signal using `librosa.effects.trim`

# Spectrogram transforms

## `SpecChannelShuffle`

_Added in v0.13.0_

Shuffle the channels of a multichannel spectrogram. This can help combat positional bias.

## `SpecFrequencyMask`

_Added in v0.13.0_

Mask a set of frequencies in a spectrogram, à la Google AI SpecAugment. This type of data
augmentation has proved to make speech recognition models more robust.

The masked frequencies can be replaced with either the mean of the original values or a
given constant (e.g. zero).

# Known limitations

* Some transforms do not support multichannel audio yet. See [Multichannel audio](#multichannel-audio)
* Expects the input dtype to be float32, and have values between -1 and 1.
* The code runs on CPU, not GPU. For a GPU-compatible version, check out [pytorch-audiomentations](https://github.com/asteroid-team/torch-audiomentations)
* Multiprocessing is not officially supported yet. See also [#46](https://github.com/iver56/audiomentations/issues/46)

Contributions are welcome!

# Multichannel audio

_The following table is valid for v0.14.0 and v0.15.0 only_

| Transform | Supports multichannel audio? |
| --------- | ---------------------------- |
| AddBackgroundNoise | - |
| AddGaussianNoise | Yes |
| AddGaussianSNR | Yes |
| AddImpulseResponse | - |
| AddShortNoises | - |
| ClippingDistortion | Yes |
| FrequencyMask | Yes |
| Gain | Yes |
| LoudnessNormalization | Yes, up to 5 channels |
| Mp3Compression | - |
| Normalize | Yes |
| PitchShift | Yes |
| PolarityInversion | Yes |
| Resample | - |
| Shift | Yes |
| SpecChannelShuffle | Yes |
| SpecFrequencyMask | Yes |
| TimeMask | Yes |
| TimeStretch | Yes |
| Trim | - |

# Version history

## v0.15.0 (2020-12-10)

* Fix picklability of instances of `AddImpulseResponse`, `AddBackgroundNoise`
 and `AddShortNoises`
* Add an option `leave_length_unchanged` to `AddImpulseResponse`

## v0.14.0 (2020-12-06)

* Implement `LoudnessNormalization`
* Implement `randomize_parameters` in `Compose`. Thanks to SolomidHero.
* Add multichannel support to `AddGaussianNoise`, `AddGaussianSNR`, `ClippingDistortion`,
`FrequencyMask`, `PitchShift`, `Shift`, `TimeMask` and `TimeStretch`

## v0.13.0 (2020-11-10)

* Show a warning if a waveform had to be resampled after loading it. This is because resampling
is slow. Ideally, files on disk should already have the desired sample rate.
* Correctly find audio files with upper case filename extensions.
* Lay the foundation for spectrogram transforms. Implement `SpecChannelShuffle` and
`SpecFrequencyMask`.
* Fix a bug where AddBackgroundNoise crashed when trying to add digital silence to an input. Thanks to juheeuu.
* Configurable LRU cache for transforms that use external sound files. Thanks to alumae.
* Officially add multichannel support to `Normalize`

## v0.12.1 (2020-09-28)

* Speed up `AddBackgroundNoise`, `AddShortNoises` and `AddImpulseResponse` by loading wav files with scipy or wavio instead of librosa.

## v0.12.0 (2020-09-23)

* Implement `Mp3Compression`
* Python <= 3.5 is no longer officially supported, since [Python 3.5 has reached end-of-life](https://devguide.python.org/#status-of-python-branches)
* Expand range of supported `librosa` versions
* Officially support multichannel audio in `Gain` and `PolarityInversion`
* Add m4a and opus to the list of recognized audio filename extensions
* Breaking change: Internal util functions are no longer exposed directly. If you were doing
    e.g. `from audiomentations import calculate_rms`, now you have to do
    `from audiomentations.core.utils import calculate_rms`


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
* Add `fade` parameter to `TimeMask`

Thanks to askskro

## v0.7.0 (2020-01-14)

Add new transforms:

* `AddGaussianSNR`
* `AddImpulseResponse`
* `FrequencyMask`
* `TimeMask`
* `Trim`

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

Audiomentations isn't the only python library that can do various types of audio data
augmentation/degradation! Here's an overview:

| Name | Github stars | License | Last commit | GPU support? |
| ---- | ------------ | ------- | ----------- | ------------ |
| [audio-degradation-toolbox](https://github.com/sevagh/audio-degradation-toolbox) | ![Github stars](https://img.shields.io/github/stars/sevagh/audio-degradation-toolbox) | ![License](https://img.shields.io/github/license/sevagh/audio-degradation-toolbox) | ![Last commit](https://img.shields.io/github/last-commit/sevagh/audio-degradation-toolbox) | ![No](https://img.shields.io/badge/GPU-No-red) |
| [audio_degrader](https://github.com/emilio-molina/audio_degrader) | ![Github stars](https://img.shields.io/github/stars/emilio-molina/audio_degrader) | ![License](https://img.shields.io/github/license/emilio-molina/audio_degrader) | ![Last commit](https://img.shields.io/github/last-commit/emilio-molina/audio_degrader) | ![No](https://img.shields.io/badge/GPU-No-red) |
| [audiomentations](https://github.com/iver56/audiomentations) | ![Github stars](https://img.shields.io/github/stars/iver56/audiomentations) | ![License](https://img.shields.io/github/license/iver56/audiomentations) | ![Last commit](https://img.shields.io/github/last-commit/iver56/audiomentations) | ![No](https://img.shields.io/badge/GPU-No-red) |
| [kapre](https://github.com/keunwoochoi/kapre) | ![Github stars](https://img.shields.io/github/stars/keunwoochoi/kapre) | ![License](https://img.shields.io/github/license/keunwoochoi/kapre) | ![Last commit](https://img.shields.io/github/last-commit/keunwoochoi/kapre) | ![Yes, Keras/Tensorflow](https://img.shields.io/badge/GPU-Keras%2FTensorflow-green) |
| [muda](https://github.com/bmcfee/muda) | ![Github stars](https://img.shields.io/github/stars/bmcfee/muda) | ![License](https://img.shields.io/github/license/bmcfee/muda) | ![Last commit](https://img.shields.io/github/last-commit/bmcfee/muda) | ![No](https://img.shields.io/badge/GPU-No-red) |
| [nlpaug](https://github.com/makcedward/nlpaug) | ![Github stars](https://img.shields.io/github/stars/makcedward/nlpaug) | ![License](https://img.shields.io/github/license/makcedward/nlpaug) | ![Last commit](https://img.shields.io/github/last-commit/makcedward/nlpaug) | ![No](https://img.shields.io/badge/GPU-No-red) |
| [pydiogment](https://github.com/SuperKogito/pydiogment) | ![Github stars](https://img.shields.io/github/stars/SuperKogito/pydiogment) | ![License](https://img.shields.io/github/license/SuperKogito/pydiogment) | ![Last commit](https://img.shields.io/github/last-commit/SuperKogito/pydiogment) | ![No](https://img.shields.io/badge/GPU-No-red) |
| [python-audio-effects](https://github.com/carlthome/python-audio-effects) | ![Github stars](https://img.shields.io/github/stars/carlthome/python-audio-effects) | ![License](https://img.shields.io/github/license/carlthome/python-audio-effects) | ![Last commit](https://img.shields.io/github/last-commit/carlthome/python-audio-effects) | ![No](https://img.shields.io/badge/GPU-No-red) |
| [sigment](https://github.com/eonu/sigment) | ![Github stars](https://img.shields.io/github/stars/eonu/sigment) | ![License](https://img.shields.io/github/license/eonu/sigment) | ![Last commit](https://img.shields.io/github/last-commit/eonu/sigment) | ![No](https://img.shields.io/badge/GPU-No-red) |
| [SpecAugment](https://github.com/DemisEom/SpecAugment) | ![Github stars](https://img.shields.io/github/stars/DemisEom/SpecAugment) | ![License](https://img.shields.io/github/license/DemisEom/SpecAugment) | ![Last commit](https://img.shields.io/github/last-commit/DemisEom/SpecAugment) | ![Yes, Pytorch & Tensorflow](https://img.shields.io/badge/GPU-Pytorch%20%26%20Tensorflow-green) |
| [spec_augment](https://github.com/zcaceres/spec_augment) | ![Github stars](https://img.shields.io/github/stars/zcaceres/spec_augment) | ![License](https://img.shields.io/github/license/zcaceres/spec_augment) | ![Last commit](https://img.shields.io/github/last-commit/zcaceres/spec_augment) | ![Yes, Pytorch](https://img.shields.io/badge/GPU-Pytorch-green) |
| [torch-audiomentations](https://github.com/asteroid-team/torch-audiomentations) | ![Github stars](https://img.shields.io/github/stars/asteroid-team/torch-audiomentations) | ![License](https://img.shields.io/github/license/asteroid-team/torch-audiomentations) | ![Last commit](https://img.shields.io/github/last-commit/asteroid-team/torch-audiomentations) | ![Yes, Pytorch](https://img.shields.io/badge/GPU-Pytorch-green) |
| [WavAugment](https://github.com/facebookresearch/WavAugment) | ![Github stars](https://img.shields.io/github/stars/facebookresearch/WavAugment) | ![License](https://img.shields.io/github/license/facebookresearch/WavAugment) | ![Last commit](https://img.shields.io/github/last-commit/facebookresearch/WavAugment) | ![Yes, Pytorch](https://img.shields.io/badge/GPU-Pytorch-green) |

# Acknowledgements

Thanks to [Nomono](https://nomono.co/) for backing audiomentations.

Thanks to [all contributors](https://github.com/iver56/audiomentations/graphs/contributors) who help improving audiomentations.
