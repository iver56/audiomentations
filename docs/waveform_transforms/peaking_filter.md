# `PeakingFilter`

_Added in v0.21.0_

Add a biquad peaking filter transform

# PeakingFilter API

[`min_center_freq`](#min_center_freq){ #min_center_freq }: `float` • unit: hertz • range: [0.0, ∞)
:   :octicons-milestone-24: Default: `50.0`. The minimum center frequency of the peaking filter

[`max_center_freq`](#max_center_freq){ #max_center_freq }: `float` • unit: hertz • range: [0.0, ∞)
:   :octicons-milestone-24: Default: `7500.0`. The maximum center frequency of the peaking filter

[`min_gain_db`](#min_gain_db){ #min_gain_db }: `float` • unit: Decibel
:   :octicons-milestone-24: Default: `-24.0`. The minimum gain at center frequency

[`max_gain_db`](#max_gain_db){ #max_gain_db }: `float` • unit: Decibel
:   :octicons-milestone-24: Default: `24.0`. The maximum gain at center frequency

[`min_q`](#min_q){ #min_q }: `float` • range: [0.0, ∞)
:   :octicons-milestone-24: Default: `0.5`. The minimum quality factor Q. The higher the
    Q, the steeper the transition band will be.

[`max_q`](#max_q){ #max_q }: `float` • range: [0.0, ∞)
:   :octicons-milestone-24: Default: `5.0`. The maximum quality factor Q. The higher the
    Q, the steeper the transition band will be.

[`p`](#p){ #p }: `float` • range: [0.0, 1.0]
:   :octicons-milestone-24: Default: `0.5`. The probability of applying this transform.
