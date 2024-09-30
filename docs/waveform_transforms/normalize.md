# `Normalize`

_Added in v0.6.0_

Apply a constant amount of gain, so that highest signal level present in the sound
becomes 0 dBFS, i.e. the loudest level allowed if all samples must be between -1 and 1.
Also known as peak normalization.

# Normalize API

[`apply_to`](#apply_to){ #apply_to }: `str` • choices: `"all"`, `"only_too_loud_sounds"`
:   :octicons-milestone-24: Default: `"all"`. Defines the criterion for applying the transform.

    * `"all"`: Apply peak normalization to all inputs
    * `"only_too_loud_sounds"`: Apply peak normalization only to inputs where the maximum absolute peak is greater than 1

[`p`](#p){ #p }: `float` • range: [0.0, 1.0]
:   :octicons-milestone-24: Default: `0.5`. The probability of applying this transform.

## Source code :octicons-mark-github-16:

[audiomentations/augmentations/normalize.py :octicons-link-external-16:](https://github.com/iver56/audiomentations/blob/main/audiomentations/augmentations/normalize.py){target=_blank}
