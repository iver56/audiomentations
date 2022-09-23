# `BandPassFilter`

_Added in v0.18.0, updated in v0.21.0_

Apply band-pass filtering to the input audio. Filter steepness (6/12/18... dB / octave)
is parametrized. Can also be set for zero-phase filtering (will result in a 6 dB drop at
cutoffs).

# BandPassFilter API

[`min_center_freq`](#min_center_freq){ #min_center_freq }: `float` (unit: hertz)
:   :octicons-milestone-24: Default: `200.0`. Minimum center frequency in hertz

[`max_center_freq`](#max_center_freq){ #max_center_freq }: `float` (unit: hertz)
:   :octicons-milestone-24: Default: `4000.0`. Maximum center frequency in hertz

[`min_bandwidth_fraction`](#min_bandwidth_fraction){ #min_bandwidth_fraction }: `float`
:   :octicons-milestone-24: Default: `0.5`. Minimum bandwidth relative to center frequency

[`max_bandwidth_fraction`](#max_bandwidth_fraction){ #max_bandwidth_fraction }: `float`
:   :octicons-milestone-24: Default: `1.99`. Maximum bandwidth relative to center frequency

[`min_rolloff`](#min_rolloff){ #min_rolloff }: `float` (unit: Decibels/octave)
:   :octicons-milestone-24: Default: `12`. Minimum filter roll-off (in dB/octave).
    Must be a multiple of 6

[`max_rolloff`](#max_rolloff){ #max_rolloff }: `float` (unit: Decibels/octave)
:   :octicons-milestone-24: Default: `24`. Maximum filter roll-off (in dB/octave)
    Must be a multiple of 6

[`zero_phase`](#zero_phase){ #zero_phase }: `bool`
:   :octicons-milestone-24: Default: `False`. Whether filtering should be zero phase.
    When this is set to `True` it will not affect the phase of the input signal but will
    sound 3 dB lower at the cutoff frequency compared to the non-zero phase case (6 dB
    vs. 3 dB). Additionally, it is 2 times slower than in the non-zero phase case. If
    you absolutely want no phase distortions (e.g. want to augment an audio file with
    lots of transients, like a drum track), set this to `True`.

[`p`](#p){ #p }: `float`
:   :octicons-milestone-24: Default: `0.5`. The probability of applying this transform.
