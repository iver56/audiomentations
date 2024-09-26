# `LoudnessNormalization`

_Added in v0.14.0_

Apply a constant amount of gain to match a specific loudness (in LUFS). This is an
implementation of ITU-R BS.1770-4.

For an explanation on LUFS, see [https://en.wikipedia.org/wiki/LUFS :octicons-link-external-16:](https://en.wikipedia.org/wiki/LUFS){target=_blank}

See also the following web pages for more info on audio loudness normalization:

* [https://github.com/csteinmetz1/pyloudnorm :octicons-link-external-16:](https://github.com/csteinmetz1/pyloudnorm){target=_blank}
* [https://en.wikipedia.org/wiki/Audio_normalization :octicons-link-external-16:](https://en.wikipedia.org/wiki/Audio_normalization){target=_blank}

Warning: This transform can return samples outside the [-1, 1] range, which may lead to
clipping or wrap distortion, depending on what you do with the audio in a later stage.
See also [https://en.wikipedia.org/wiki/Clipping_(audio)#Digital_clipping :octicons-link-external-16:](https://en.wikipedia.org/wiki/Clipping_(audio)#Digital_clipping)

# LoudnessNormalization API

~~[`min_lufs_in_db`](#min_lufs_in_db){ #min_lufs_in_db }: `float` • unit: LUFS~~
:   :warning: Deprecated as of v0.31.0, removed as of v0.38.0. Use [`min_lufs`](#min_lufs) instead

~~[`max_lufs_in_db`](#max_lufs_in_db){ #max_lufs_in_db }: `float` • unit: LUFS~~
:   :warning: Deprecated as of v0.31.0, removed as of v0.38.0. Use [`max_lufs`](#max_lufs) instead

[`min_lufs`](#min_lufs){ #min_lufs }: `float` • unit: LUFS
:   :octicons-milestone-24: Default: `-31.0`. Minimum loudness target.

[`max_lufs`](#max_lufs){ #max_lufs }: `float` • unit: LUFS
:   :octicons-milestone-24: Default: `-13.0`. Maximum loudness target.

[`p`](#p){ #p }: `float` • range: [0.0, 1.0]
:   :octicons-milestone-24: Default: `0.5`. The probability of applying this transform.

## Source code :octicons-mark-github-16:

[audiomentations/augmentations/loudness_normalization.py :octicons-link-external-16:](https://github.com/iver56/audiomentations/blob/main/audiomentations/augmentations/loudness_normalization.py){target=_blank}
