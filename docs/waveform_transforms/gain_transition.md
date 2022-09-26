# `GainTransition`

_Added in v0.22.0_

Gradually change the volume up or down over a random time span. Also known as
fade in and fade out. The fade works on a logarithmic scale, which is natural to
human hearing.

The way this works is that it picks two gains: a first gain and a second gain.
Then it picks a time range for the transition between those two gains.
Note that this transition can start before the audio starts and/or end after the
audio ends, so the output audio can start or end in the middle of a transition.
The gain starts at the first gain and is held constant until the transition start.
Then it transitions to the second gain. Then that gain is held constant until the
end of the sound.

# GainTransition API

[`min_gain_in_db`](#min_gain_in_db){ #min_gain_in_db }: `float` (unit: Decibel)
:   :octicons-milestone-24: Default: `-24.0`. Minimum gain.

[`max_gain_in_db`](#max_gain_in_db){ #max_gain_in_db }: `float` (unit: Decibel)
:   :octicons-milestone-24: Default: `6.0`. Maximum gain.

[`min_duration`](#min_duration){ #min_duration }: `Union[float, int]` (unit: see [`duration_unit`](#duration_unit))
:   :octicons-milestone-24: Default: `0.2`. Minimum length of transition.

[`max_duration`](#max_duration){ #max_duration }: `Union[float, int]` (unit: see [`duration_unit`](#duration_unit))
:   :octicons-milestone-24: Default: `6.0`. Maximum length of transition.

[`duration_unit`](#duration_unit){ #duration_unit }: `str` (choices: `"fraction"`, `"samples"`, `"seconds"`)
:   :octicons-milestone-24: Default: `"seconds"`. Defines the unit of the value of `min_duration` and `max_duration`.
    
    * `"fraction"`: Fraction of the total sound length
    * `"samples"`: Number of audio samples
    * `"seconds"`: Number of seconds

[`p`](#p){ #p }: `float` (range: [0.0, 1.0])
:   :octicons-milestone-24: Default: `0.5`. The probability of applying this transform.
