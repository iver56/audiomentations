# `ClippingDistortion`

_Added in v0.8.0_

Distort signal by clipping a random percentage of points

The percentage of points that will be clipped is drawn from a uniform distribution between
the two input parameters `min_percentile_threshold` and `max_percentile_threshold`. If for instance
30% is drawn, the samples are clipped if they're below the 15th or above the 85th percentile.

# ClippingDistortion API

[`min_percentile_threshold`](#min_percentile_threshold){ #min_percentile_threshold }: `int`
:   :octicons-milestone-24: Default: `0`. A lower bound on the total percent of samples
    that will be clipped

[`max_percentile_threshold`](#max_percentile_threshold){ #max_percentile_threshold }: `int`
:   :octicons-milestone-24: Default: `40`. An upper bound on the total percent of
    samples that will be clipped

[`p`](#p){ #p }: `float`
:   :octicons-milestone-24: Default: `0.5`. The probability of applying this transform.
