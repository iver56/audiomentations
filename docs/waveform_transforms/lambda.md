# `Lambda`

_Added in v0.26.0_

Apply a user-defined transform (callable) to the signal. The inspiration for this
transform comes from albumentation's lambda transform. This allows one to have a little
more fine-grained control over the operations in the context of a `Compose`, `OneOf` or `SomeOf`

## Usage example

```python
import random

from audiomentations import Lambda, OneOf, Gain


def gain_only_left_channel(samples, sample_rate):
    samples[0, :] *= random.uniform(0.8, 1.25)
    return samples


transform = OneOf(
    transforms=[Lambda(transform=gain_only_left_channel, p=1.0), Gain(p=1.0)]
)

augmented_sound = transform(my_stereo_waveform_ndarray, sample_rate=16000)
```

# Lambda API

[`transform`](#transform){ #transform }: `Callable`
:   :octicons-milestone-24: A callable to be applied. It should input
    samples (ndarray), sample_rate (int) and optionally some user-defined
    keyword arguments.

[`p`](#p){ #p }: `float` • range: [0.0, 1.0]
:   :octicons-milestone-24: Default: `0.5`. The probability of applying this transform.

[`**kwargs`](#**kwargs){ #**kwargs }
:   :octicons-milestone-24: Optional extra parameters passed to the callable transform

## Source code :octicons-mark-github-16:

[audiomentations/augmentations/lambda_transform.py :octicons-link-external-16:](https://github.com/iver56/audiomentations/blob/main/audiomentations/augmentations/lambda_transform.py){target=_blank}
