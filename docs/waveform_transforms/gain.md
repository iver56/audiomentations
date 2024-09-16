# `Gain`

_Added in v0.11.0_

Multiply the audio by a random amplitude factor to reduce or increase the volume. This
technique can help a model become somewhat invariant to the overall gain of the input audio.

Warning: This transform can return samples outside the [-1, 1] range, which may lead to
clipping or wrap distortion, depending on what you do with the audio in a later stage.
See also [https://en.wikipedia.org/wiki/Clipping_(audio)#Digital_clipping](https://en.wikipedia.org/wiki/Clipping_(audio)#Digital_clipping)

## Input-output example

Here we input a speech recording and apply a -8 dB gain.

![Input-output waveforms and spectrograms](Gain.webp)

| Input sound                                                                 | Transformed sound                                                                 |
|-----------------------------------------------------------------------------|-----------------------------------------------------------------------------------|
| <audio controls><source src="../Gain_input.flac" type="audio/flac"></audio> | <audio controls><source src="../Gain_transformed.flac" type="audio/flac"></audio> | 


# Gain API

[`min_gain_in_db`](#min_gain_in_db){ #min_gain_in_db }: `float` • unit: Decibel
:   :warning: Deprecated as of v0.31.0, removed as of v0.38.0. Use [`min_gain_db`](#min_gain_db) instead

[`max_gain_in_db`](#max_gain_in_db){ #max_gain_in_db }: `float` • unit: Decibel
:   :warning: Deprecated as of v0.31.0, removed as of v0.38.0. Use [`max_gain_db`](#max_gain_db) instead

[`min_gain_db`](#min_gain_db){ #min_gain_db }: `float` • unit: Decibel
:   :octicons-milestone-24: Default: `-12.0`. Minimum gain.

[`max_gain_db`](#max_gain_db){ #max_gain_db }: `float` • unit: Decibel
:   :octicons-milestone-24: Default: `12.0`. Maximum gain.

[`p`](#p){ #p }: `float` • range: [0.0, 1.0]
:   :octicons-milestone-24: Default: `0.5`. The probability of applying this transform.

## Source code :octicons-mark-github-16:

[audiomentations/augmentations/gain.py :octicons-link-external-16:](https://github.com/iver56/audiomentations/blob/main/audiomentations/augmentations/gain.py){target=_blank}
