# `Gain`

_Added in v0.11.0_

Multiply the audio by a random amplitude factor to reduce or increase the volume. This
technique can help a model become somewhat invariant to the overall gain of the input audio.

Warning: This transform can return samples outside the [-1, 1] range, which may lead to
clipping or wrap distortion, depending on what you do with the audio in a later stage.
See also [https://en.wikipedia.org/wiki/Clipping_(audio)#Digital_clipping](https://en.wikipedia.org/wiki/Clipping_(audio)#Digital_clipping)

# Gain API

[`min_gain_in_db`](#min_gain_in_db){ #min_gain_in_db }: `float` (unit: Decibel)
:   :octicons-milestone-24: Default: `-12.0`. Minimum gain.

[`max_gain_in_db`](#max_gain_in_db){ #max_gain_in_db }: `float` (unit: Decibel)
:   :octicons-milestone-24: Default: `12.0`. Maximum gain.

[`p`](#p){ #p }: `float`
:   :octicons-milestone-24: Default: `0.5`. The probability of applying this transform.
