# `Padding`

_Added in v0.23.0_

Apply padding to the audio signal - take a fraction of the end or the start of the
audio and replace that part with padding. This can be useful for preparing ML models
with constant input length for padded inputs.

# Padding API

[`mode`](#mode){ #mode }: `str` • choices: `"silence"`, `"wrap"`, `"reflect"`
:   :octicons-milestone-24: Default: `"silence"`. Padding mode.

[`min_fraction`](#min_fraction){ #min_fraction }: `float` • range: [0.0, 1.0]
:   :octicons-milestone-24: Default: `0.01`. Minimum fraction of the signal duration to be padded

[`max_fraction`](#max_fraction){ #max_fraction }: `float` • range: [0.0, 1.0]
:   :octicons-milestone-24: Default: `0.7`. Maximum fraction of the signal duration to be padded

[`pad_section`](#pad_section){ #pad_section }: `str` • choices: `"start"`, `"end"`
:   :octicons-milestone-24: Default: `"end"`. Which part of the signal should be replaced with padding

[`p`](#p){ #p }: `float` • range: [0.0, 1.0]
:   :octicons-milestone-24: Default: `0.5`. The probability of applying this transform.
