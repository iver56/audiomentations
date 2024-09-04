# `Normalize`

_Added in v0.6.0_

Apply a constant amount of gain, so that highest signal level present in the sound
becomes 0 dBFS, i.e. the loudest level allowed if all samples must be between -1 and 1.
Also known as peak normalization.

# Normalize API

[`p`](#p){ #p }: `float` • range: [0.0, 1.0]
:   :octicons-milestone-24: Default: `0.5`. The probability of applying this transform.

## Source code :octicons-mark-github-16:

[audiomentations/augmentations/normalize.py :octicons-link-external-16:](https://github.com/iver56/audiomentations/blob/main/audiomentations/augmentations/normalize.py){target=_blank}
