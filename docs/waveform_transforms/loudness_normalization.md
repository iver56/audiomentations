# `LoudnessNormalization`

_Added in v0.14.0_

Apply a constant amount of gain to match a specific loudness. This is an implementation of
ITU-R BS.1770-4.

Warning: This transform can return samples outside the [-1, 1] range, which may lead to
clipping or wrap distortion, depending on what you do with the audio in a later stage.
See also https://en.wikipedia.org/wiki/Clipping_(audio)#Digital_clipping
