# `Resample`

_Added in v0.8.0_

Resample signal using librosa.core.resample

To do downsampling only, set both minimum and maximum sampling rate lower than original
sampling rate. Conversely, to perform upsampling only, set both rates higher than the original sampling rate.

# Resample API

[`min_sample_rate`](#min_sample_rate){ #min_sample_rate }: `int` • unit: Hz
:   :octicons-milestone-24: Default: `8000`. The minimum sample rate

[`max_sample_rate`](#max_sample_rate){ #max_sample_rate }: `int` • unit: Hz
:   :octicons-milestone-24: Default: `44100`. The maximum sample rate

[`p`](#p){ #p }: `float` • range: [0.0, 1.0]
:   :octicons-milestone-24: Default: `0.5`. The probability of applying this transform.

## Source code :octicons-mark-github-16:

[audiomentations/augmentations/resample.py :octicons-link-external-16:](https://github.com/iver56/audiomentations/blob/main/audiomentations/augmentations/resample.py){target=_blank}
