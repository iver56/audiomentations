# `LowPassFilter`

_Added in v0.18.0, updated in v0.21.0_

Apply low-pass filtering to the input audio of parametrized filter steepness (6/12/18... dB / octave).
Can also be set for zero-phase filtering (will result in a 6 dB drop at cutoff).

# LowPassFilter API

[`min_cutoff_freq`](#min_cutoff_freq){ #min_cutoff_freq }: `float` • unit: hertz
:   :octicons-milestone-24: Default: `150.0`. Minimum cutoff frequency

[`max_cutoff_freq`](#max_cutoff_freq){ #max_cutoff_freq }: `float` • unit: hertz
:   :octicons-milestone-24: Default: `7500.0`. Maximum cutoff frequency

[`min_rolloff`](#min_rolloff){ #min_rolloff }: `int` • unit: Decibels/octave
:   :octicons-milestone-24: Default: `12`. Minimum filter roll-off (in dB/octave).
    Must be a multiple of 6 (or 12 if `zero_phase` is `True`)

[`max_rolloff`](#max_rolloff){ #max_rolloff }: `int` • unit: Decibels/octave
:   :octicons-milestone-24: Default: `24`. Maximum filter roll-off (in dB/octave)
    Must be a multiple of 6 (or 12 if `zero_phase` is `True`)

[`zero_phase`](#zero_phase){ #zero_phase }: `bool`
:   :octicons-milestone-24: Default: `False`. Whether filtering should be zero phase.
    When this is set to `True`, it will not affect the phase of the input signal but will
    sound 3 dB lower at the cutoff frequency compared to the non-zero phase case (6 dB
    vs. 3 dB). Additionally, it is twice as slow as the non-zero phase case. If
    you absolutely want no phase distortions (e.g. want to augment an audio file with
    lots of transients, like a drum track), set this to `True`.

[`p`](#p){ #p }: `float` • range: [0.0, 1.0]
:   :octicons-milestone-24: Default: `0.5`. The probability of applying this transform.

## Source code :octicons-mark-github-16:

[audiomentations/augmentations/low_pass_filter.py :octicons-link-external-16:](https://github.com/iver56/audiomentations/blob/main/audiomentations/augmentations/low_pass_filter.py){target=_blank}
