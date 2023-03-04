# `AirAbsorption`

_Added in v0.25.0_

A lowpass-like filterbank with variable octave attenuation that simulates attenuation of
high frequencies due to air absorption. This transform is parametrized by temperature,
humidity, and the distance between audio source and microphone.

This is not a scientifically accurate transform but basically applies a uniform
filterbank with attenuations given by:

`att = exp(- distance * absorption_coefficient)`

where `distance` is the microphone-source assumed distance in meters and `absorption_coefficient`
is adapted from a lookup table by [pyroomacoustics](https://github.com/LCAV/pyroomacoustics).
It can also be seen as a lowpass filter with variable octave attenuation.

Note: This only "simulates" the dampening of high frequencies, and does not
attenuate according to the distance law. Gain augmentation needs to be done separately.

## AirAbsorption API

[`min_temperature`](#min_temperature){ #min_temperature }: `float` • unit: Celsius • choices: [10.0, 20.0]
:   :octicons-milestone-24: Default: `10.0`. Minimum temperature in Celsius (can take a value of either 10.0 or 20.0)

[`max_temperature`](#max_temperature){ #max_temperature }: `float` • unit: Celsius • choices: [10.0, 20.0]
:   :octicons-milestone-24: Default: `20.0`. Maximum temperature in Celsius (can take a value of either 10.0 or 20.0)

[`min_humidity`](#min_humidity){ #min_humidity }: `float` • unit: percent • range: [30.0, 90.0]
:   :octicons-milestone-24: Default: `30.0`. Minimum humidity in percent (between 30.0 and 90.0)

[`max_humidity`](#max_humidity){ #max_humidity }: `float` • unit: percent • range: [30.0, 90.0]
:   :octicons-milestone-24: Default: `90.0`. Maximum humidity in percent (between 30.0 and 90.0)

[`min_distance`](#min_distance){ #min_distance }: `float` • unit: meters
:   :octicons-milestone-24: Default: `10.0`. Minimum microphone-source distance in meters.

[`max_distance`](#max_distance){ #max_distance }: `float` • unit: meters
:   :octicons-milestone-24: Default: `100.0`. Maximum microphone-source distance in meters.

[`p`](#p){ #p }: `float` • range: [0.0, 1.0]
:   :octicons-milestone-24: Default: `0.5`. The probability of applying this transform.
