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
:   :octicons-milestone-24: Default: `0.01`. Minimum length of the silent part as a
    fraction of the total sound length.

[`max_band_part`](#max_band_part){ #max_band_part }: `float` • range: [0.0, 1.0]
:   :octicons-milestone-24: Default: `0.2`. Maximum length of the silent part as a
    fraction of the total sound length.

[`fade_duration`](#fade_duration){ #fade_duration }: `float` • unit: seconds • range: 0.0 or [0.00025, ∞)
: :octicons-milestone-24: Default: `0.005`. Duration of the fade-in and fade-out applied
    at the edges of the silent region to smooth transitions and avoid abrupt
    changes, which can otherwise produce impulses or clicks in the audio.
    If you need hard edges or clicks, set this to `0.0` to disable fading.
    Positive values must be at least 0.00025.

[`mask_location`](#mask_location){ #mask_location }: `str` • choices: `"start"`, `"end"`, `"random"`
: :octicons-milestone-24: Default: `random`. Where to place the silent region.
    
    * `"start"`: silence begins at index 0
    * `"end"`: silence ends at the last sample
    * `"random"`: silence starts at a random position

[`p`](#p){ #p }: `float` • range: [0.0, 1.0]
:   :octicons-milestone-24: Default: `0.5`. The probability of applying this transform.

## Old TimeMask API (<=v0.40.0)

:warning: This only applies to version 0.40.0 and older

[`fade`](#fade){ #fade }: `bool`
:   :octicons-milestone-24: Default: `False`. When set to `True`, a linear fade-in and fade-out is added to the silent part.
    This can smooth out unwanted abrupt changes between consecutive samples, which might
    otherwise sound like transients/clicks/pops.

## Source code :octicons-mark-github-16:

[audiomentations/augmentations/time_mask.py :octicons-link-external-16:](https://github.com/iver56/audiomentations/blob/main/audiomentations/augmentations/time_mask.py){target=_blank}
