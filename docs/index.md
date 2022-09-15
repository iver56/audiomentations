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

| Feature                 | Extra dependencies                  |
|-------------------------|-------------------------------------|
| `Limiter`               | `cylimiter`                         |
| `LoudnessNormalization` | `pyloudnorm`                        |
| `Mp3Compression`        | `ffmpeg` and [`pydub` or `lameenc`] |
| `RoomSimulator`         | `pyroomacoustics`                   |

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

Some of the waveform transforms can be visualized (for understanding) by the [audio-transformation-visualization GUI](https://share.streamlit.io/phrasenmaeher/audio-transformation-visualization/main/visualize_transformation.py) (made by phrasenmaeher), where you can upload your own input wav file

For a list and brief explanation of all waveform transforms, see Waveform transforms

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

# Acknowledgements

Thanks to [Nomono](https://nomono.co/) for backing audiomentations.

Thanks to [all contributors](https://github.com/iver56/audiomentations/graphs/contributors) who help improving audiomentations.
