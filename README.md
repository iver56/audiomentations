# Audiomentations

[![Build status](https://img.shields.io/circleci/project/github/iver56/audiomentations/master.svg)](https://circleci.com/gh/iver56/audiomentations)
[![Code coverage](https://img.shields.io/codecov/c/github/iver56/audiomentations/master.svg)](https://codecov.io/gh/iver56/audiomentations)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/ambv/black)
[![Licence: MIT](https://img.shields.io/pypi/l/audiomentations)](https://github.com/iver56/audiomentations/blob/master/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6367011.svg)](https://doi.org/10.5281/zenodo.6367011)

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

Here are some examples of datasets that can be downloaded and used as background noise:

* https://github.com/karolpiczak/ESC-50#download
* https://github.com/microsoft/DNS-Challenge/

## `AddGaussianNoise`

_Added in v0.1.0_

Add gaussian noise to the samples

## `AddGaussianSNR`

_Added in v0.7.0_

Add gaussian noise to the input. A random Signal to Noise Ratio (SNR) will be picked
uniformly in the decibel scale. This aligns with human hearing, which is more
logarithmic than linear.

## `ApplyImpulseResponse`

_Added in v0.7.0_ 

Convolve the audio with a random impulse response.
Impulse responses can be created using e.g. http://tulrich.com/recording/ir_capture/

Some datasets of impulse responses are publicly available:
- [EchoThief](http://www.echothief.com/) containing 115 impulse responses acquired in a wide range of locations.
- [The MIT McDermott](https://mcdermottlab.mit.edu/Reverb/IR_Survey.html) dataset containing 271 impulse responses acquired in everyday places.

Impulse responses are represented as wav files in the given ir_path.

## `AddShortNoises`

_Added in v0.9.0_

Mix in various (bursts of overlapping) sounds with random pauses between. Useful if your
original sound is clean and you want to simulate an environment where short noises sometimes
occur.

A folder of (noise) sounds to be mixed in must be specified.

## `BandPassFilter`

_Added in v0.18.0, updated in v0.21.0_

Apply band-pass filtering to the input audio. Filter steepness (6/12/18... dB / octave)
is parametrized. Can also be set for zero-phase filtering (will result in a 6db drop at
cutoffs).

## `BandStopFilter`

_Added in v0.21.0_

Apply band-stop filtering to the input audio. Also known as notch filter or
band reject filter. It relates to the frequency mask idea in the SpecAugment paper.
This transform is similar to FrequencyMask, but has overhauled default parameters
and parameter randomization - center frequency gets picked in mel space so it is
more aligned with human hearing, which is not linear. Filter steepness
(6/12/18... dB / octave) is parametrized. Can also be set for zero-phase filtering
(will result in a 6db drop at cutoffs).

## `Clip`

_Added in v0.17.0_

Clip audio by specified values. e.g. set a_min=-1.0 and a_max=1.0 to ensure that no
samples in the audio exceed that extent. This can be relevant for avoiding integer
overflow or underflow (which results in unintended wrap distortion that can sound
horrible) when exporting to e.g. 16-bit PCM wav.

Another way of ensuring that all values stay between -1.0 and 1.0 is to apply
`PeakNormalization`.

This transform is different from `ClippingDistortion` in that it takes fixed values
for clipping instead of clipping a random percentile of the samples. Arguably, this
transform is not very useful for data augmentation. Instead, think of it as a very
cheap and harsh limiter (for samples that exceed the allotted extent) that can
sometimes be useful at the end of a data augmentation pipeline.

## `ClippingDistortion`

_Added in v0.8.0_

Distort signal by clipping a random percentage of points

The percentage of points that will be clipped is drawn from a uniform distribution between
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

## `GainTransition`

_Added in v0.22.0_

Gradually change the volume up or down over a random time span. Also known as
fade in and fade out. The fade works on a logarithmic scale, which is natural to
human hearing.

## `HighPassFilter`

_Added in v0.18.0, updated in v0.21.0_

Apply high-pass filtering to the input audio of parametrized filter steepness (6/12/18... dB / octave).
Can also be set for zero-phase filtering (will result in a 6db drop at cutoff).

## `HighShelfFilter`

_Added in v0.21.0_

A high shelf filter is a filter that either boosts (increases amplitude) or cuts
(decreases amplitude) frequencies above a certain center frequency. This transform
applies a high-shelf filter at a specific center frequency in hertz.
The gain at nyquist frequency is controlled by `{min,max}_gain_db` (note: can be positive or negative!).
Filter coefficients are taken from [the W3 Audio EQ Cookbook](https://www.w3.org/TR/audio-eq-cookbook/)

## `LowPassFilter`

_Added in v0.18.0, updated in v0.21.0_

Apply low-pass filtering to the input audio of parametrized filter steepness (6/12/18... dB / octave).
Can also be set for zero-phase filtering (will result in a 6db drop at cutoff).

## `LowShelfFilter`

_Added in v0.21.0_

A low shelf filter is a filter that either boosts (increases amplitude) or cuts
(decreases amplitude) frequencies below a certain center frequency. This transform
applies a low-shelf filter at a specific center frequency in hertz.
The gain at DC frequency is controlled by `{min,max}_gain_db` (note: can be positive or negative!).
Filter coefficients are taken from [the W3 Audio EQ Cookbook](https://www.w3.org/TR/audio-eq-cookbook/)

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

## `Padding`

_Added in v0.23.0_

Apply padding to the audio signal - take a fraction of the end or the start of the
audio and replace that part with padding. This can be useful for preparing ML models
with constant input length for padded inputs.

## `PeakingFilter`

_Added in v0.21.0_

Add a biquad peaking filter transform

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

## `Reverse`

_Added in v0.18.0_

Reverse the audio. Also known as time inversion. Inversion of an audio track along its time
axis relates to the random flip of an image, which is an augmentation technique that is
widely used in the visual domain. This can be relevant in the context of audio
classification. It was successfully applied in the paper
[AudioCLIP: Extending CLIP to Image, Text and Audio](https://arxiv.org/pdf/2106.13043.pdf).

## `RoomSimulator`

_Added in v0.23.0_

A ShoeBox Room Simulator. Simulates a cuboid of parametrized size and average surface absorption coefficient. It also includes a source
and microphones in parametrized locations.

Use it when you want a ton of synthetic room impulse responses of specific configurations
characteristics or simply to quickly add reverb for augmentation purposes

## `SevenBandParametricEQ`

_Added in v0.24.0_

Adjust the volume of different frequency bands. This transform is a 7-band
parametric equalizer - a combination of one low shelf filter, five peaking filters
and one high shelf filter, all with randomized gains, Q values and center frequencies.

Because this transform changes the timbre, but keeps the overall "class" of the
sound the same (depending on application), it can be used for data augmentation to
make ML models more robust to various frequency spectrums. Many things can affect
the spectrum, like room acoustics, any objects between the microphone and
the sound source, microphone type/model and the distance between the sound source
and the microphone.

The seven bands have center frequencies picked in the following ranges (min-max):
42-95 hz
91-204 hz
196-441 hz
421-948 hz
909-2045 hz
1957-4404 hz
4216-9486 hz

## `Shift`

_Added in v0.5.0_

Shift the samples forwards or backwards, with or without rollover

## `TanhDistortion`

_Added in v0.19.0_

Apply tanh (hyperbolic tangent) distortion to the audio. This technique is sometimes
used for adding distortion to guitar recordings. The tanh() function can give a rounded
"soft clipping" kind of distortion, and the distortion amount is proportional to the
loudness of the input and the pre-gain. Tanh is symmetric, so the positive and
negative parts of the signal are squashed in the same way. This transform can be
useful as data augmentation because it adds harmonics. In other words, it changes
the timbre of the sound.

See this page for examples: [http://gdsp.hf.ntnu.no/lessons/3/17/](http://gdsp.hf.ntnu.no/lessons/3/17/)

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

* Some transforms do not support multichannel audio yet. See [Multichannel audio](#multichannel-audio)
* Expects the input dtype to be float32, and have values between -1 and 1.
* The code runs on CPU, not GPU. For a GPU-compatible version, check out [pytorch-audiomentations](https://github.com/asteroid-team/torch-audiomentations)
* Multiprocessing is not officially supported yet. See also [#46](https://github.com/iver56/audiomentations/issues/46)

Contributions are welcome!

# Multichannel audio

As of v0.22.0, all transforms except `AddBackgroundNoise` and `AddShortNoises` support not only mono audio (1-dimensional numpy arrays), but also stereo audio, i.e. 2D arrays with shape like `(num_channels, num_samples)`

# Changelog

## Unreleased

## v0.24.0 (2022-03-18)

### Added

* Add `SevenBandParametricEQ` transform
* Add optional `noise_transform` in `AddShortNoises`
* Add .aac and .aif to the list of recognized audio filename endings

### Changed

* Show warning if `top_db` and/or `p` in `Trim` are not specified because their default
 values will change in a future version

### Fixed

* Fix filter instability bug related to center freq above nyquist freq in `LowShelfFilter` and `HighShelfFilter`

## v0.23.0 (2022-03-07)

### Added

* Add `Padding` transform
* Add `RoomSimulator` transform for simulating shoebox rooms using `pyroomacoustics`
* Add parameter `signal_gain_in_db_during_noise` in `AddShortNoises`

### Changed

* Not specifying a value for `leave_length_unchanged` in `AddImpulseResponse` now emits
 a warning, as the default value will change from `False` to `True` in a future version.

### Removed

* Remove the deprecated `AddImpulseResponse` alias. Use `ApplyImpulseResponse` instead.
* Remove support for the legacy parameters `min_SNR` and `max_SNR` in `AddGaussianSNR`
* Remove useless default path value in `AddBackgroundNoise`, `AddShortNoises` and `ApplyImpulseResponse`

## v0.22.0 (2022-02-18)

### Added

* Implement `GainTransition`
* Add support for librosa 0.9
* Add support for stereo audio in `Mp3Compression`, `Resample` and `Trim`
* Add `"relative_to_whole_input"` option for `noise_rms` parameter in `AddShortNoises`
* Add optional `noise_transform` in `AddBackgroundNoise`

### Changed

* Improve speed of `PitchShift` by 6-18% when the input audio is stereo

### Removed

* Remove support for librosa<=0.7.2

## v0.21.0 (2022-02-10)

### Added

* Add support for multichannel audio in `ApplyImpulseResponse`, `BandPassFilter`, `HighPassFilter` and `LowPassFilter`
* Add `BandStopFilter` (similar to FrequencyMask, but with overhauled defaults and parameter randomization behavior), `PeakingFilter`, `LowShelfFilter` and `HighShelfFilter`
* Add parameter `add_all_noises_with_same_level` in `AddShortNoises`

### Changed

* Change `BandPassFilter`, `LowPassFilter`, `HighPassFilter`, to use scipy's butterworth
 filters instead of pydub. Now they have parametrized roll-off. Filters are now steeper
 than before by default - set `min_rolloff=6, max_rolloff=6` to get the old behavior.
 They also support zero-phase filtering now. And they're at least ~25x times faster than before! 

### Removed

* Remove optional `wavio` dependency for audio loading

## v0.20.0 (2021-11-18)

### Added

* Implement `OneOf` and `SomeOf` for applying one of or some of many transforms. Transforms are randomly
 chosen every call. Inspired by augly. Thanks to Cangonin and iver56.
* Add a new argument `apply_to_children` (bool) in `randomize_parameters`,
 `freeze_parameters` and `unfreeze_parameters` in `Compose` and `SpecCompose`.

### Changed

* Insert three new parameters in `AddBackgroundNoise`: `noise_rms` (defaults to "relative", which is 
 the old behavior), `min_absolute_rms_in_db` and `max_absolute_rms_in_db`. This **may be a breaking
 change** if you used `AddBackgroundNoise` with positional arguments in earlier versions of audiomentations!
 Please use keyword arguments to be on the safe side - it should be backwards compatible then.

### Fixed

* Remove global `pydub` import which was accidentally introduced in v0.18.0. `pydub` is
 considered an optional dependency and is imported only on demand now.

## v0.19.0 (2021-10-18)

### Added

* Implement `TanhDistortion`. Thanks to atamazian and iver56.
* Add a `noise_rms` parameter to `AddShortNoises`. It defaults to `relative`, which
 is the old behavior. `absolute` allows for adding loud noises to parts that are
 relatively silent in the input.

## v0.18.0 (2021-08-05)

### Added

* Implement `BandPassFilter`, `HighPassFilter`, `LowPassFilter` and `Reverse`. Thanks to atamazian.

## v0.17.0 (2021-06-25)

### Added

* Add a `fade` option in `Shift` for eliminating unwanted clicks
* Add support for 32-bit int wav loading with scipy>=1.6
* Add support for float64 wav files. However, the use of this format is discouraged,
  since float32 is more than enough for audio in most cases.
* Implement `Clip`. Thanks to atamazian.
* Add some parameter sanity checks in `AddGaussianNoise`
* Officially support librosa 0.8.1

### Changed

* Rename `AddImpulseResponse` to `ApplyImpulseResponse`. The former will still work for
  now, but give a warning.
* When looking for audio files in `AddImpulseResponse`, `AddBackgroundNoise`
  and `AddShortNoises`, follow symlinks by default.
* When using the new parameters `min_snr_in_db` and `max_snr_in_db` in `AddGaussianSNR`,
  SNRs will be picked uniformly in _the decibel scale_ instead of in the linear amplitude
  ratio scale. The new behavior aligns more with human hearing, which is not linear.

### Fixed

* Avoid division by zero in `AddImpulseResponse` when input is digital silence (all zeros)
* Fix inverse SNR characteristics in `AddGaussianSNR`. It will continue working as before
  unless you switch to the new parameters `min_snr_in_db` and `max_snr_in_db`. If you
  use the old parameters, you'll get a warning.

## v0.16.0 (2021-02-11)

### Added

* Implement `SpecCompose` for applying a pipeline of spectrogram transforms. Thanks to omerferhatt.

### Fixed

* Fix a bug in `SpecChannelShuffle` where it did not support more than 3 audio channels. Thanks to omerferhatt.
* Limit scipy version range to >=1.0,<1.6 to avoid issues with loading 24-bit wav files.
Support for scipy>=1.6 will be added later.

## v0.15.0 (2020-12-10)

### Added

* Add an option `leave_length_unchanged` to `AddImpulseResponse`

### Fixed

* Fix picklability of instances of `AddImpulseResponse`, `AddBackgroundNoise`
 and `AddShortNoises`

## v0.14.0 (2020-12-06)

### Added

* Implement `LoudnessNormalization`
* Implement `randomize_parameters` in `Compose`. Thanks to SolomidHero.
* Add multichannel support to `AddGaussianNoise`, `AddGaussianSNR`, `ClippingDistortion`,
`FrequencyMask`, `PitchShift`, `Shift`, `TimeMask` and `TimeStretch`

## v0.13.0 (2020-11-10)

### Added

* Lay the foundation for spectrogram transforms. Implement `SpecChannelShuffle` and
`SpecFrequencyMask`.
* Configurable LRU cache for transforms that use external sound files. Thanks to alumae.
* Officially add multichannel support to `Normalize`

### Changed

* Show a warning if a waveform had to be resampled after loading it. This is because resampling
is slow. Ideally, files on disk should already have the desired sample rate.

### Fixed

* Correctly find audio files with upper case filename extensions.
* Fix a bug where AddBackgroundNoise crashed when trying to add digital silence to an input. Thanks to juheeuu.

## v0.12.1 (2020-09-28)

### Changed

* Speed up `AddBackgroundNoise`, `AddShortNoises` and `AddImpulseResponse` by loading wav files with scipy or wavio instead of librosa.

## v0.12.0 (2020-09-23)

### Added

* Implement `Mp3Compression`
* Officially support multichannel audio in `Gain` and `PolarityInversion`
* Add m4a and opus to the list of recognized audio filename extensions

### Changed

* Expand range of supported `librosa` versions

### Removed

* Python <= 3.5 is no longer officially supported, since [Python 3.5 has reached end-of-life](https://devguide.python.org/#status-of-python-branches)
* Breaking change: Internal util functions are no longer exposed directly. If you were doing
    e.g. `from audiomentations import calculate_rms`, now you have to do
    `from audiomentations.core.utils import calculate_rms`

## v0.11.0 (2020-08-27)

### Added

* Implement `Gain` and `PolarityInversion`. Thanks to Spijkervet for the inspiration.

## v0.10.1 (2020-07-27)

### Changed

* Improve the performance of `AddBackgroundNoise` and `AddShortNoises` by optimizing the implementation of `calculate_rms`.

### Fixed

* Improve compatibility of output files written by the demo script. Thanks to xwJohn.
* Fix division by zero bug in `Normalize`. Thanks to ZFTurbo.

## v0.10.0 (2020-05-05)

### Added

* `AddImpulseResponse`, `AddBackgroundNoise` and `AddShortNoises` now support aiff files in addition to flac, mp3, ogg and wav

### Changed

* Breaking change: `AddImpulseResponse`, `AddBackgroundNoise` and `AddShortNoises` now include subfolders when searching for files. This is useful when your sound files are organized in subfolders.

### Fixed

* Fix filter instability bug in `FrequencyMask`. Thanks to kvilouras.

## v0.9.0 (2020-02-20)

### Added

* Remember randomized/chosen effect parameters. This allows for freezing the parameters and applying the same effect to multiple sounds. Use transform.freeze_parameters() and transform.unfreeze_parameters() for this.
* Implement transform.serialize_parameters(). Useful for when you want to store metadata on how a sound was perturbed.
* Add a rollover parameter to `Shift`. This allows for introducing silence instead of a wrapped part of the sound.
* Add support for flac in `AddImpulseResponse`
* Implement `AddBackgroundNoise` transform. Useful for when you want to add background noise to all of your sound. You need to give it a folder of background noises to choose from.
* Implement `AddShortNoises`. Useful for when you want to add (bursts of) short noise sounds to your input audio.

### Changed

* Disregard non-audio files when looking for impulse response files
* Switch to a faster convolve implementation. This makes `AddImpulseResponse` significantly faster.
* Expand supported range of librosa versions

### Fixed

* Fix a bug in `ClippingDistortion` where the min_percentile_threshold was not respected as expected.
* Improve handling of empty input

## v0.8.0 (2020-01-28)

### Added

* Add shuffle parameter in `Composer`
* Add `Resample` transformation
* Add `ClippingDistortion` transformation
* Add `fade` parameter to `TimeMask`

Thanks to askskro

## v0.7.0 (2020-01-14)

### Added

* `AddGaussianSNR`
* `AddImpulseResponse`
* `FrequencyMask`
* `TimeMask`
* `Trim`

Thanks to karpnv

## v0.6.0 (2019-05-27)

### Added

* Implement peak normalization

## v0.5.0 (2019-02-23)

### Added

* Implement `Shift` transform

### Changed

* Ensure p is within bounds

## v0.4.0 (2019-02-19)

### Added

* Implement `PitchShift` transform

### Fixed

* Fix output dtype of `AddGaussianNoise`

## v0.3.0 (2019-02-19)

### Added

* Implement `leave_length_unchanged` in `TimeStretch`

## v0.2.0 (2019-02-18)

### Added

* Add `TimeStretch` transform
* Parametrize `AddGaussianNoise`

## v0.1.0 (2019-02-15)

### Added

* Initial release. Includes only one transform: `AddGaussianNoise`


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
