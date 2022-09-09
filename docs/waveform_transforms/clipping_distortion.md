# `ClippingDistortion`

_Added in v0.8.0_

Distort signal by clipping a random percentage of points

The percentage of points that will be clipped is drawn from a uniform distribution between
the two input parameters min_percentile_threshold and max_percentile_threshold. If for instance
30% is drawn, the samples are clipped if they're below the 15th or above the 85th percentile.
