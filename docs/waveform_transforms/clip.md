# `Clip`

_Added in v0.17.0_

Clip audio by specified values. e.g. set `a_min=-1.0` and `a_max=1.0` to ensure that no
samples in the audio exceed that extent. This can be relevant for avoiding integer
overflow or underflow (which results in unintended wrap distortion that can sound
horrible) when exporting to e.g. 16-bit PCM wav.

Another way of ensuring that all values stay between -1.0 and 1.0 is to apply
`PeakNormalization`.

This transform is different from `ClippingDistortion` in that it takes fixed values
for clipping instead of clipping a random percentile of the samples. Arguably, this
transform is not very useful for data augmentation. Instead, think of it as a very
cheap and harsh limiter (for samples that exceed the allotted extent) that can
sometimes be useful at the end of a data augmentation pipeline.

# Clip API

[`a_min`](#a_min){ #a_min }: `float`
:   :octicons-milestone-24: Default: `-1.0`. Minimum value for clipping.

[`a_max`](#a_max){ #a_max }: `float`
:   :octicons-milestone-24: Default: `1.0`. Maximum value for clipping.

[`p`](#p){ #p }: `float` (range: [0.0, 1.0])
:   :octicons-milestone-24: Default: `0.5`. The probability of applying this transform.
