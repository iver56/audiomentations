# Audiomentations

[![Build status](https://img.shields.io/circleci/project/github/iver56/audiomentations/master.svg)](https://circleci.com/gh/iver56/audiomentations)
[![Code coverage](https://img.shields.io/codecov/c/github/iver56/audiomentations/master.svg)](https://codecov.io/gh/iver56/audiomentations)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/ambv/black)
[![Licence: MIT](https://img.shields.io/pypi/l/audiomentations)](https://github.com/iver56/audiomentations/blob/master/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7010042.svg)](https://doi.org/10.5281/zenodo.7010042)

A Python library for audio data augmentation. Inspired by
[albumentations](https://github.com/albu/albumentations). Useful for deep learning. Runs on
CPU. Supports mono audio and [multichannel audio](#multichannel-audio). Can be
integrated in training pipelines in e.g. Tensorflow/Keras or Pytorch. Has helped people get
world-class results in Kaggle competitions. Is used by companies making next-generation audio
products.

Need a Pytorch-specific alternative with GPU support? Check out [torch-audiomentations](https://github.com/asteroid-team/torch-audiomentations)!

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
| `LoudnessNormalization` | `pyloudnorm` |
| `Mp3Compression` | `ffmpeg` and [`pydub` or `lameenc`] |
| `RoomSimulator` | `pyroomacoustics` |

Note: `ffmpeg` can be installed via e.g. conda or from [the official ffmpeg download page](http://ffmpeg.org/download.html).

# Usage example

## Waveform

```python
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import numpy as np

augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
])

# Generate 2 seconds of dummy audio for the sake of example
samples = np.random.uniform(low=-0.2, high=0.2, size=(32000,)).astype(np.float32)

# Augment/transform/perturb the audio data
augmented_samples = augment(samples=samples, sample_rate=16000)
```

Check out the source code at [audiomentations/augmentations/](https://github.com/iver56/audiomentations/blob/master/audiomentations/augmentations/) to see the waveform transforms you can apply, and what arguments they have.

## Spectrogram

```python
from audiomentations import SpecCompose, SpecChannelShuffle, SpecFrequencyMask
import numpy as np

augment = SpecCompose(
    [
        SpecChannelShuffle(p=0.5),
        SpecFrequencyMask(p=0.5),
    ]
)

# Example spectrogram with 1025 frequency bins, 256 time steps and 2 audio channels
spectrogram = np.random.random((1025, 256, 2))

# Augment/transform/perturb the spectrogram
augmented_spectrogram = augment(spectrogram)
```

See [audiomentations/spec_augmentations/spectrogram_transforms.py](https://github.com/iver56/audiomentations/blob/master/audiomentations/spec_augmentations/spectrogram_transforms.py) for spectrogram transforms.

# Waveform transforms

Some of the following waveform transforms can be visualized (for understanding) by the [audio-transformation-visualization GUI](https://share.streamlit.io/phrasenmaeher/audio-transformation-visualization/main/visualize_transformation.py) (made by phrasenmaeher), where you can upload your own input wav file

For a list and brief explanation of all waveform transforms, see [Waveform transforms](waveform_transforms.md)

# Spectrogram transforms

For a list and brief explanation of all spectrogram transforms, see [Spectrogram transforms](spectrogram_transforms.md)

# Composition classes

## `Compose`

Compose applies the given sequence of transforms when called, optionally shuffling the sequence for every call.

## `SpecCompose`

Same as Compose, but for spectrogram transforms

## `OneOf`

OneOf randomly picks one of the given transforms when called, and applies that transform.

## `SomeOf`

SomeOf randomly picks several of the given transforms when called, and applies those transforms.

# Known limitations

* A few transforms do not support multichannel audio yet. See [Multichannel audio](#multichannel-audio)
* Expects the input dtype to be float32, and have values between -1 and 1.
* The code runs on CPU, not GPU. For a GPU-compatible version, check out [pytorch-audiomentations](https://github.com/asteroid-team/torch-audiomentations)
* Multiprocessing probably works but is not _officially_ supported yet

Contributions are welcome!

# Multichannel audio

As of v0.22.0, all transforms except `AddBackgroundNoise` and `AddShortNoises` support not only mono audio (1-dimensional numpy arrays), but also stereo audio, i.e. 2D arrays with shape like `(num_channels, num_samples)`

# Alternatives

Audiomentations isn't the only python library that can do various types of audio data
augmentation/degradation! Here's an overview:

| Name | Github stars | License | Last commit | GPU support? |
| ---- | ------------ | ------- | ----------- | ------------ |
| [audio-degradation-toolbox](https://github.com/sevagh/audio-degradation-toolbox) | ![Github stars](https://img.shields.io/github/stars/sevagh/audio-degradation-toolbox) | ![License](https://img.shields.io/github/license/sevagh/audio-degradation-toolbox) | ![Last commit](https://img.shields.io/github/last-commit/sevagh/audio-degradation-toolbox) | ![No](https://img.shields.io/badge/GPU-No-red) |
| [audio_degrader](https://github.com/emilio-molina/audio_degrader) | ![Github stars](https://img.shields.io/github/stars/emilio-molina/audio_degrader) | ![License](https://img.shields.io/github/license/emilio-molina/audio_degrader) | ![Last commit](https://img.shields.io/github/last-commit/emilio-molina/audio_degrader) | ![No](https://img.shields.io/badge/GPU-No-red) |
| [audiomentations](https://github.com/iver56/audiomentations) | ![Github stars](https://img.shields.io/github/stars/iver56/audiomentations) | ![License](https://img.shields.io/github/license/iver56/audiomentations) | ![Last commit](https://img.shields.io/github/last-commit/iver56/audiomentations) | ![No](https://img.shields.io/badge/GPU-No-red) |
| [AugLy](https://github.com/facebookresearch/AugLy/tree/main/augly/audio) | ![Github stars](https://img.shields.io/github/stars/facebookresearch/AugLy) | ![License](https://img.shields.io/github/license/facebookresearch/AugLy) | ![Last commit](https://img.shields.io/github/last-commit/facebookresearch/AugLy) | ![No](https://img.shields.io/badge/GPU-No-red) |
| [kapre](https://github.com/keunwoochoi/kapre) | ![Github stars](https://img.shields.io/github/stars/keunwoochoi/kapre) | ![License](https://img.shields.io/github/license/keunwoochoi/kapre) | ![Last commit](https://img.shields.io/github/last-commit/keunwoochoi/kapre) | ![Yes, Keras/Tensorflow](https://img.shields.io/badge/GPU-Keras%2FTensorflow-green) |
| [muda](https://github.com/bmcfee/muda) | ![Github stars](https://img.shields.io/github/stars/bmcfee/muda) | ![License](https://img.shields.io/github/license/bmcfee/muda) | ![Last commit](https://img.shields.io/github/last-commit/bmcfee/muda) | ![No](https://img.shields.io/badge/GPU-No-red) |
| [nlpaug](https://github.com/makcedward/nlpaug) | ![Github stars](https://img.shields.io/github/stars/makcedward/nlpaug) | ![License](https://img.shields.io/github/license/makcedward/nlpaug) | ![Last commit](https://img.shields.io/github/last-commit/makcedward/nlpaug) | ![No](https://img.shields.io/badge/GPU-No-red) |
| [pedalboard](https://github.com/spotify/pedalboard) | ![Github stars](https://img.shields.io/github/stars/spotify/pedalboard) | ![License](https://img.shields.io/github/license/spotify/pedalboard) | ![Last commit](https://img.shields.io/github/last-commit/spotify/pedalboard) | ![No](https://img.shields.io/badge/GPU-No-red) |
| [pydiogment](https://github.com/SuperKogito/pydiogment) | ![Github stars](https://img.shields.io/github/stars/SuperKogito/pydiogment) | ![License](https://img.shields.io/github/license/SuperKogito/pydiogment) | ![Last commit](https://img.shields.io/github/last-commit/SuperKogito/pydiogment) | ![No](https://img.shields.io/badge/GPU-No-red) |
| [python-audio-effects](https://github.com/carlthome/python-audio-effects) | ![Github stars](https://img.shields.io/github/stars/carlthome/python-audio-effects) | ![License](https://img.shields.io/github/license/carlthome/python-audio-effects) | ![Last commit](https://img.shields.io/github/last-commit/carlthome/python-audio-effects) | ![No](https://img.shields.io/badge/GPU-No-red) |
| [sigment](https://github.com/eonu/sigment) | ![Github stars](https://img.shields.io/github/stars/eonu/sigment) | ![License](https://img.shields.io/github/license/eonu/sigment) | ![Last commit](https://img.shields.io/github/last-commit/eonu/sigment) | ![No](https://img.shields.io/badge/GPU-No-red) |
| [SpecAugment](https://github.com/DemisEom/SpecAugment) | ![Github stars](https://img.shields.io/github/stars/DemisEom/SpecAugment) | ![License](https://img.shields.io/github/license/DemisEom/SpecAugment) | ![Last commit](https://img.shields.io/github/last-commit/DemisEom/SpecAugment) | ![Yes, Pytorch & Tensorflow](https://img.shields.io/badge/GPU-Pytorch%20%26%20Tensorflow-green) |
| [spec_augment](https://github.com/zcaceres/spec_augment) | ![Github stars](https://img.shields.io/github/stars/zcaceres/spec_augment) | ![License](https://img.shields.io/github/license/zcaceres/spec_augment) | ![Last commit](https://img.shields.io/github/last-commit/zcaceres/spec_augment) | ![Yes, Pytorch](https://img.shields.io/badge/GPU-Pytorch-green) |
| [teal](https://github.com/am1tyadav/teal) | ![Github stars](https://img.shields.io/github/stars/am1tyadav/teal) | ![License](https://img.shields.io/github/license/am1tyadav/teal) | ![Last commit](https://img.shields.io/github/last-commit/am1tyadav/teal) | ![Yes, Keras/Tensorflow](https://img.shields.io/badge/GPU-Keras%2FTensorflow-green) |
| [torch-audiomentations](https://github.com/asteroid-team/torch-audiomentations) | ![Github stars](https://img.shields.io/github/stars/asteroid-team/torch-audiomentations) | ![License](https://img.shields.io/github/license/asteroid-team/torch-audiomentations) | ![Last commit](https://img.shields.io/github/last-commit/asteroid-team/torch-audiomentations) | ![Yes, Pytorch](https://img.shields.io/badge/GPU-Pytorch-green) |
| [torchaudio-augmentations](https://github.com/Spijkervet/torchaudio-augmentations) | ![Github stars](https://img.shields.io/github/stars/Spijkervet/torchaudio-augmentations) | ![License](https://img.shields.io/github/license/Spijkervet/torchaudio-augmentations) | ![Last commit](https://img.shields.io/github/last-commit/Spijkervet/torchaudio-augmentations) | ![Yes, Pytorch](https://img.shields.io/badge/GPU-Pytorch-green) |
| [WavAugment](https://github.com/facebookresearch/WavAugment) | ![Github stars](https://img.shields.io/github/stars/facebookresearch/WavAugment) | ![License](https://img.shields.io/github/license/facebookresearch/WavAugment) | ![Last commit](https://img.shields.io/github/last-commit/facebookresearch/WavAugment) | ![No](https://img.shields.io/badge/GPU-No-red) |

# Acknowledgements

Thanks to [Nomono](https://nomono.co/) for backing audiomentations.

Thanks to [all contributors](https://github.com/iver56/audiomentations/graphs/contributors) who help improving audiomentations.
