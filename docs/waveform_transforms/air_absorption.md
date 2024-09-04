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

Note that since this transform mostly affects high frequencies, it is only
suitable for audio with sufficiently high sample rate, like 32 kHz and above.

Note also that this transform only "simulates" the dampening of high frequencies, and
does not attenuate according to the distance law. Gain augmentation needs to be done
separately.

## Input-output example

Here we input a high-quality speech recording and apply `AirAbsorption` with an air
temperature of 20 degrees celsius, 70% humidity and a distance of 20 meters. One can see
clearly in the spectrogram that the highs, especially above ~13 kHz, are rolled off in
the output, but it may require a quiet room and some concentration to
hear it clearly in the audio comparison.

![Input-output waveforms and spectrograms](AirAbsorption.webp)

| Input sound                                                                           | Transformed sound                                                                           |
|---------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| <audio controls><source src="../AirAbsorption_input.flac" type="audio/flac"></audio> | <audio controls><source src="../AirAbsorption_transformed.flac" type="audio/flac"></audio> | 

## Usage example

```python
from audiomentations import AirAbsorption

transform = AirAbsorption(
    min_distance=10.0,
    max_distance=50.0,
    p=1.0,
)

augmented_sound = transform(my_waveform_ndarray, sample_rate=48000)
```

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

## Source code :octicons-mark-github-16:

[audiomentations/augmentations/air_absorption.py :octicons-link-external-16:](https://github.com/iver56/audiomentations/blob/main/audiomentations/augmentations/air_absorption.py){target=_blank}
