# `SevenBandParametricEQ`

_Added in v0.24.0_

Adjust the volume of different frequency bands. This transform is a 7-band
parametric equalizer - a combination of one low shelf filter, five peaking filters
and one high shelf filter, all with randomized gains, Q values and center frequencies.

Because this transform changes the timbre, but keeps the overall "class" of the
sound the same (depending on application), it can be used for data augmentation to
make ML models more robust to various frequency spectrums. Many things can affect
the spectrum, for example:

* the nature and quality of the sound source
* room acoustics
* any objects between the microphone and the sound source
* microphone type/model
* the distance between the sound source and the microphone

The seven bands have center frequencies picked in the following ranges (min-max):

* 42-95 Hz
* 91-204 Hz
* 196-441 Hz
* 421-948 Hz
* 909-2045 Hz
* 1957-4404 Hz
* 4216-9486 Hz


## SevenBandParametricEQ API

[`min_gain_db`](#min_gain_db){ #min_gain_db }: `float` • unit: Decibel
:   :octicons-milestone-24: Default: `-12.0`. Minimum number of dB to cut or boost a band

[`max_gain_db`](#max_gain_db){ #max_gain_db }: `float` • unit: Decibel
:   :octicons-milestone-24: Default: `12.0`. Maximum number of dB to cut or boost a band

[`p`](#p){ #p }: `float` • range: [0.0, 1.0]
:   :octicons-milestone-24: Default: `0.5`. The probability of applying this transform.

## Source code :octicons-mark-github-16:

[audiomentations/augmentations/seven_band_parametric_eq.py :octicons-link-external-16:](https://github.com/iver56/audiomentations/blob/main/audiomentations/augmentations/seven_band_parametric_eq.py){target=_blank}
