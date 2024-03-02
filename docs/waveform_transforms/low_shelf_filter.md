# `LowShelfFilter`

_Added in v0.21.0_

A low shelf filter is a filter that either boosts (increases amplitude) or cuts
(decreases amplitude) frequencies below a certain center frequency. This transform
applies a low-shelf filter at a specific center frequency in hertz.
The gain at DC frequency is controlled by `{min,max}_gain_db` (note: can be positive or negative!).
Filter coefficients are taken from [the W3 Audio EQ Cookbook :octicons-link-external-16:](https://www.w3.org/TR/audio-eq-cookbook/)

# LowShelfFilter API

[`min_center_freq`](#min_center_freq){ #min_center_freq }: `float` • unit: hertz
:   :octicons-milestone-24: Default: `50.0`. The minimum center frequency of the shelving filter

[`max_center_freq`](#max_center_freq){ #max_center_freq }: `float` • unit: hertz
:   :octicons-milestone-24: Default: `4000.0`. The maximum center frequency of the shelving filter

[`min_gain_db`](#min_gain_db){ #min_gain_db }: `float` • unit: Decibel
:   :octicons-milestone-24: Default: `-18.0`. The minimum gain at DC (0 Hz)

[`max_gain_db`](#max_gain_db){ #max_gain_db }: `float` • unit: Decibel
:   :octicons-milestone-24: Default: `18.0`. The maximum gain at DC (0 Hz)

[`min_q`](#min_q){ #min_q }: `float` • range: (0.0, 1.0]
:   :octicons-milestone-24: Default: `0.1`. The minimum quality factor Q. The higher
    the Q, the steeper the transition band will be.

[`max_q`](#max_q){ #max_q }: `float` • range: (0.0, 1.0]
:   :octicons-milestone-24: Default: `0.999`. The maximum quality factor Q. The higher
    the Q, the steeper the transition band will be.

[`p`](#p){ #p }: `float` • range: [0.0, 1.0]
:   :octicons-milestone-24: Default: `0.5`. The probability of applying this transform.
