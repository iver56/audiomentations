# Audiomentations

[![Build status](https://img.shields.io/circleci/project/github/iver56/audiomentations/master.svg)](https://circleci.com/gh/iver56/audiomentations)
[![Code coverage](https://img.shields.io/codecov/c/github/iver56/audiomentations/master.svg)](https://codecov.io/gh/iver56/audiomentations)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/ambv/black)
[![Licence: MIT](https://img.shields.io/pypi/l/audiomentations)](https://github.com/iver56/audiomentations/blob/master/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7074562.svg)](https://doi.org/10.5281/zenodo.7074562)

A Python library for audio data augmentation. Inspired by
[albumentations](https://github.com/albu/albumentations). Useful for deep learning. Runs on
CPU. Supports mono audio and multichannel audio. Can be
integrated in training pipelines in e.g. Tensorflow/Keras or Pytorch. Has helped people get
world-class results in Kaggle competitions. Is used by companies making next-generation audio
products.

Need a Pytorch-specific alternative with GPU support? Check out [torch-audiomentations](https://github.com/asteroid-team/torch-audiomentations)!

# Setup

![Python version support](https://img.shields.io/pypi/pyversions/audiomentations)
[![PyPI version](https://img.shields.io/pypi/v/audiomentations.svg?style=flat)](https://pypi.org/project/audiomentations/)
[![Number of downloads from PyPI per month](https://img.shields.io/pypi/dm/audiomentations.svg?style=flat)](https://pypi.org/project/audiomentations/)

`pip install audiomentations`

# Usage example

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

# Documentation

See [https://iver56.github.io/audiomentations/](https://iver56.github.io/audiomentations/)

# Transforms

* [AddBackgroundNoise](https://iver56.github.io/audiomentations/waveform_transforms/add_background_noise/)
* [AddGaussianNoise](https://iver56.github.io/audiomentations/waveform_transforms/add_gaussian_noise/)
* [AddGaussianSNR](https://iver56.github.io/audiomentations/waveform_transforms/add_gaussian_snr/)
* [AddShortNoises](https://iver56.github.io/audiomentations/waveform_transforms/add_short_noises/)
* [AirAbsorption](https://iver56.github.io/audiomentations/waveform_transforms/air_absorption/)
* [ApplyImpulseResponse](https://iver56.github.io/audiomentations/waveform_transforms/apply_impulse_response/)
* [BandPassFilter](https://iver56.github.io/audiomentations/waveform_transforms/band_pass_filter/)
* [BandStopFilter](https://iver56.github.io/audiomentations/waveform_transforms/band_stop_filter/)
* [Clip](https://iver56.github.io/audiomentations/waveform_transforms/clip/)
* [ClippingDistortion](https://iver56.github.io/audiomentations/waveform_transforms/clipping_distortion/)
* [Gain](https://iver56.github.io/audiomentations/waveform_transforms/gain/)
* [GainTransition](https://iver56.github.io/audiomentations/waveform_transforms/gain_transition/)
* [HighPassFilter](https://iver56.github.io/audiomentations/waveform_transforms/high_pass_filter/)
* [HighShelfFilter](https://iver56.github.io/audiomentations/waveform_transforms/high_shelf_filter/)
* [Lambda](https://iver56.github.io/audiomentations/waveform_transforms/lambda/)
* [Limiter](https://iver56.github.io/audiomentations/waveform_transforms/limiter/)
* [LoudnessNormalization](https://iver56.github.io/audiomentations/waveform_transforms/loudness_normalization/)
* [LowPassFilter](https://iver56.github.io/audiomentations/waveform_transforms/low_pass_filter/)
* [LowShelfFilter](https://iver56.github.io/audiomentations/waveform_transforms/low_shelf_filter/)
* [Mp3Compression](https://iver56.github.io/audiomentations/waveform_transforms/mp3_compression/)
* [Normalize](https://iver56.github.io/audiomentations/waveform_transforms/normalize/)
* [Padding](https://iver56.github.io/audiomentations/waveform_transforms/padding/)
* [PeakingFilter](https://iver56.github.io/audiomentations/waveform_transforms/peaking_filter/)
* [PitchShift](https://iver56.github.io/audiomentations/waveform_transforms/pitch_shift/)
* [PolarityInversion](https://iver56.github.io/audiomentations/waveform_transforms/polarity_inversion/)
* [Resample](https://iver56.github.io/audiomentations/waveform_transforms/resample/)
* [Reverse](https://iver56.github.io/audiomentations/waveform_transforms/reverse/)
* [RoomSimulator](https://iver56.github.io/audiomentations/waveform_transforms/room_simulator/)
* [SevenBandParametricEQ](https://iver56.github.io/audiomentations/waveform_transforms/seven_band_parametric_eq/)
* [Shift](https://iver56.github.io/audiomentations/waveform_transforms/shift/)
* [SpecChannelShuffle](https://iver56.github.io/audiomentations/spectrogram_transforms/)
* [SpecFrequencyMask](https://iver56.github.io/audiomentations/spectrogram_transforms/)
* [TanhDistortion](https://iver56.github.io/audiomentations/waveform_transforms/tanh_distortion/)
* [TimeMask](https://iver56.github.io/audiomentations/waveform_transforms/time_mask/)
* [TimeStretch](https://iver56.github.io/audiomentations/waveform_transforms/time_stretch/)
* [Trim](https://iver56.github.io/audiomentations/waveform_transforms/trim/)

# Changelog

See [https://iver56.github.io/audiomentations/changelog/](https://iver56.github.io/audiomentations/changelog/)

# Acknowledgements

Thanks to [Nomono](https://nomono.co/) for backing audiomentations.

Thanks to [all contributors](https://github.com/iver56/audiomentations/graphs/contributors) who help improving audiomentations.
