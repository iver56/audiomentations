# `Lambda`

_Added in v0.26.0_

Apply a user-defined transform (callable) to the signal.

# Lambda API

[`transform`](#transform){ #transform }: `Callable`
:   :octicons-milestone-24: A callable to be applied. It should input
    samples (ndarray), sample_rate (int) and optionally some user-defined
    keyword arguments.

[`p`](#p){ #p }: `float` • range: [0.0, 1.0]
:   :octicons-milestone-24: Default: `0.5`. The probability of applying this transform.

[`**kwargs`](#**kwargs){ #**kwargs }
:   :octicons-milestone-24: Optional extra parameters passed to the callable transform
