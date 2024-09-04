# `TimeMask`

_Added in v0.7.0_

Make a randomly chosen part of the audio silent. Inspired by
[https://arxiv.org/pdf/1904.08779.pdf](https://arxiv.org/pdf/1904.08779.pdf)


## Input-output example

Here we silence a part of a speech recording.

![Input-output waveforms and spectrograms](TimeMask.webp)

| Input sound                                                                               | Transformed sound                                                                               |
|-------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------|
| <audio controls><source src="../TimeMask_input.flac" type="audio/flac"></audio> | <audio controls><source src="../TimeMask_transformed.flac" type="audio/flac"></audio> | 


## Usage example

```python
from audiomentations import TimeMask

transform = TimeMask(
    min_band_part=0.1,
    max_band_part=0.15,
    fade=True,
    p=1.0,
)

augmented_sound = transform(my_waveform_ndarray, sample_rate=16000)
```

## TimeMask API

[`min_band_part`](#min_band_part){ #min_band_part }: `float` • range: [0.0, 1.0]
:   :octicons-milestone-24: Default: `0.0`. Minimum length of the silent part as a
    fraction of the total sound length.

[`max_band_part`](#max_band_part){ #max_band_part }: `float` • range: [0.0, 1.0]
:   :octicons-milestone-24: Default: `0.5`. Maximum length of the silent part as a
    fraction of the total sound length.

[`fade`](#fade){ #fade }: `bool`
:   :octicons-milestone-24: Default: `False`. When set to `True`, add a linear fade in
    and fade out of the silent part. This can smooth out an unwanted abrupt change
    between two consecutive samples (which sounds like a transient/click/pop).

[`p`](#p){ #p }: `float` • range: [0.0, 1.0]
:   :octicons-milestone-24: Default: `0.5`. The probability of applying this transform.

## Source code :octicons-mark-github-16:

[audiomentations/augmentations/time_mask.py :octicons-link-external-16:](https://github.com/iver56/audiomentations/blob/main/audiomentations/augmentations/time_mask.py){target=_blank}
