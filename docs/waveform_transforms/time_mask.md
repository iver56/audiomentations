# `TimeMask`

_Added in v0.7.0_

Make a randomly chosen part of the audio silent. Inspired by
[https://arxiv.org/pdf/1904.08779.pdf](https://arxiv.org/pdf/1904.08779.pdf)

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
