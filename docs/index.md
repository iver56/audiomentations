# Audiomentations documentation

[![Build status](https://img.shields.io/circleci/project/github/iver56/audiomentations/main.svg)](https://circleci.com/gh/iver56/audiomentations)
[![Code coverage](https://img.shields.io/codecov/c/github/iver56/audiomentations/main.svg)](https://codecov.io/gh/iver56/audiomentations)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/ambv/black)
[![Licence: MIT](https://img.shields.io/pypi/l/audiomentations)](https://github.com/iver56/audiomentations/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17112730.svg)](https://doi.org/10.5281/zenodo.17112730)

Audiomentations is a Python library for audio data augmentation, built to be fast and easy to use - its API is inspired by
[albumentations](https://github.com/albu/albumentations). It's useful for making audio deep learning models work well in the real world, not just in the lab.
Audiomentations runs on CPU, supports mono audio and [multichannel audio](#multichannel-audio) and integrates well in training pipelines,
such as those built with TensorFlow/Keras or PyTorch. It has helped users achieve
world-class results in Kaggle competitions and is trusted by companies building next-generation audio products with AI.

Need a Pytorch-specific alternative with GPU support? Check out [torch-audiomentations](https://github.com/asteroid-team/torch-audiomentations)!

# Setup

![Python version support](https://img.shields.io/pypi/pyversions/audiomentations)
[![PyPI version](https://img.shields.io/pypi/v/audiomentations.svg?style=flat)](https://pypi.org/project/audiomentations/)
[![Number of downloads from PyPI per month](https://img.shields.io/pypi/dm/audiomentations.svg?style=flat)](https://pypi.org/project/audiomentations/)

`pip install audiomentations`

## Optional requirements

Some features have extra dependencies. Extra python package dependencies can be installed by running

`pip install audiomentations[extras]`

| Feature                 | Extra dependencies    |
|-------------------------|-----------------------|
| `Limiter`               | `numpy-audio-limiter` |
| `LoudnessNormalization` | `loudness`            |
| `Mp3Compression`        | `fast-mp3-augment`    |
| `RoomSimulator`         | `pyroomacoustics`     |

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
    Shift(p=0.5),
])

# Generate 2 seconds of dummy audio for the sake of example
samples = np.random.uniform(low=-0.2, high=0.2, size=(32000,)).astype(np.float32)

# Augment/transform/perturb the audio data
augmented_samples = augment(samples=samples, sample_rate=16000)
```

# Waveform transforms

For a list and explanation of all waveform transforms, see Waveform transforms in the menu.

Waveform transforms can be visualized (for understanding) by the [audio-transformation-visualization GUI](https://share.streamlit.io/phrasenmaeher/audio-transformation-visualization/main/visualize_transformation.py) (made by [phrasenmaeher](https://github.com/phrasenmaeher)), where you can upload your own input wav file

# Composition classes

## `Compose`

Compose applies the given sequence of transforms when called, optionally shuffling the sequence for every call.

## `OneOf`

OneOf randomly picks one of the given transforms when called, and applies that transform. 

An optional `weights` list of floats may be given to guide the probability of each transform for being chosen. If not specified, a transform is chosen uniformly at random.

Code example:

```python
from audiomentations import OneOf, PitchShift

pitch_shift = OneOf(
    transforms=[
        PitchShift(method="librosa_phase_vocoder"),
        PitchShift(method="signalsmith_stretch"),
    ],
    p=1.0,
    weights=[0.1, 0.9],
)
```

## `SomeOf`

SomeOf randomly picks several of the given transforms when called, and applies those transforms.

# Known limitations

* A few transforms do not support multichannel audio yet. See [Multichannel audio](#multichannel-audio)
* Expects the input dtype to be float32, and have values between -1 and 1.
* The code runs on CPU, not GPU. For a GPU-compatible version, check out [pytorch-audiomentations](https://github.com/asteroid-team/torch-audiomentations)
* Multiprocessing probably works but is not _officially_ supported yet

Contributions are welcome!

# Multichannel audio

As of v0.22.0, all transforms except `AddBackgroundNoise` and `AddShortNoises` support not only mono audio (1-dimensional numpy arrays), but also stereo audio, i.e. 2D arrays with shape like `(num_channels, num_samples)`. See also the [guide on multichannel audio array shapes](guides/multichannel_audio_array_shapes.md).

# Acknowledgements

Thanks to [Nomono](https://nomono.co/) for backing audiomentations.

Thanks to [all contributors](https://github.com/iver56/audiomentations/graphs/contributors) who help improving audiomentations.
