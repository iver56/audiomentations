# `Shift`

_Added in v0.5.0_

Shift the samples forwards or backwards, with or without rollover

## Shift API

:information_source: This only applies to version 0.33.0 and newer. If you are using an older
version, you should consider upgrading. Or if you really want to keep using the old
version, you can check the ["Old Shift API (<=v0.32.0)" section](#old-shift-api-v0320) below

[`min_shift`](#min_shift){ #min_shift }: `float | int`
:   :octicons-milestone-24: Default: `-0.5`. Minimum amount of shifting in time. See also
    [`shift_unit`](#shift_unit).

[`max_shift`](#max_shift){ #max_shift }: `float | int`
:   :octicons-milestone-24: Default: `0.5`. Maximum amount of shifting in time. See also
    [`shift_unit`](#shift_unit).

[`shift_unit`](#shift_unit){ #shift_unit }: `str` • choices: `"fraction"`, `"samples"`, `"seconds"`
:   :octicons-milestone-24: Default: `"fraction"` Defines the unit of the value of
    [`min_shift`](#min_shift) and [`max_shift`](#max_shift).

    * `"fraction"`: Fraction of the total sound length
    * `"samples"`: Number of audio samples
    * `"seconds"`: Number of seconds

[`rollover`](#rollover){ #rollover }: `bool`
:   :octicons-milestone-24: Default: `True`. When set to `True`, samples that roll
    beyond the first or last position are re-introduced at the last or first. When set
    to `False`, samples that roll beyond the first or last position are discarded. In
    other words, `rollover=False` results in an empty space (with zeroes).

[`fade_duration`](#fade_duration){ #fade_duration }: `float` • unit: seconds • range: 0.0 or [0.00025, ∞)
:   :octicons-milestone-24: Default: `0.005`. If you set this to a positive number,
    there will be a fade in and/or out at the "stitch" (that was the start or the end
    of the audio before the shift). This can smooth out an unwanted abrupt
    change between two consecutive samples (which sounds like a
    transient/click/pop). This parameter denotes the duration of the fade in
    seconds. To disable the fading feature, set this parameter to 0.0.

[`p`](#p){ #p }: `float` • range: [0.0, 1.0]
:   :octicons-milestone-24: Default: `0.5`. The probability of applying this transform.

## Old Shift API (<=v0.32.0)

:warning: This only applies to version 0.32.0 and older

`min_fraction`: `float` • range: [-1, 1]
:   :octicons-milestone-24: Default: `-0.5`. Minimum fraction of total sound length to
    shift.

`max_fraction`: `float` • range: [-1, 1]
:   :octicons-milestone-24: Default: `0.5`. Maximum fraction of total sound length to
    shift.

`rollover`: `bool`
:   :octicons-milestone-24: Default: `True`. When set to `True`, samples that roll beyond the
    last position are re-introduced at the first position, and samples that roll beyond the first
    position are re-introduced at the last position. When set to `False`, samples that roll
    beyond the first or last position are discarded. In other words, `rollover=False` results
    in an empty space (with zeroes).

`fade`: `bool`
:   :octicons-milestone-24: Default: `False`. When set to `True`, there will be a short
    fade in and/or out at the "stitch" (that was the start or the end of the audio
    before the shift). This can smooth out an unwanted abrupt change between two
    consecutive samples, which would otherwise sound like a transient/click/pop.

`fade_duration`: `float` • unit: seconds
:   :octicons-milestone-24: Default: `0.01`. If `fade=True`, then this is the duration
    of the fade in seconds.

`p`: `float` • range: [0.0, 1.0]
:   :octicons-milestone-24: Default: `0.5`. The probability of applying this transform.

## Source code :octicons-mark-github-16:

[audiomentations/augmentations/shift.py :octicons-link-external-16:](https://github.com/iver56/audiomentations/blob/main/audiomentations/augmentations/shift.py){target=_blank}
