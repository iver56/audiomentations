# `Shift`

_Added in v0.5.0_

Shift the samples forwards or backwards, with or without rollover


## Shift API

[`min_fraction`](#min_fraction){ #min_fraction }: `float` • range: [-1, 1]
:   :octicons-milestone-24: Default: `-0.5`. Minimum fraction of total sound length to
    shift.

[`max_fraction`](#max_fraction){ #max_fraction }: `float` • range: [-1, 1]
:   :octicons-milestone-24: Default: `0.5`. Maximum fraction of total sound length to
    shift.

[`rollover`](#rollover){ #rollover }: `bool`
:   :octicons-milestone-24: Default: `True`. When set to `True`, samples that roll
    beyond the first or last position are re-introduced at the last or first. When set
    to `False`, samples that roll beyond the first or last position are discarded. In
    other words, `rollover=False` results in an empty space (with zeroes).

[`fade`](#fade){ #fade }: `bool`
:   :octicons-milestone-24: Default: `False`. When set to `True`, there will be a short
    fade in and/or out at the "stitch" (that was the start or the end of the audio
    before the shift). This can smooth out an unwanted abrupt change between two
    consecutive samples (which sounds like a transient/click/pop).

[`fade_duration`](#fade_duration){ #fade_duration }: `float` • unit: seconds
:   :octicons-milestone-24: Default: `0.01`. If `fade=True`, then this is the duration
    of the fade in seconds.

[`p`](#p){ #p }: `float` • range: [0.0, 1.0]
:   :octicons-milestone-24: Default: `0.5`. The probability of applying this transform.
