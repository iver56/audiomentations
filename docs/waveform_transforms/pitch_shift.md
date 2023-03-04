# `PitchShift`

_Added in v0.4.0_

Pitch shift the sound up or down without changing the tempo

# PitchShift API

[`min_semitones`](#min_semitones){ #min_semitones }: `float` • unit: semitones • range: [-12.0, 12.0]
:   :octicons-milestone-24: Default: `-4.0`. Minimum semitones to shift. Negative number means shift down.

[`max_semitones`](#max_semitones){ #max_semitones }: `float` • unit: semitones • range: [-12.0, 12.0]
:   :octicons-milestone-24: Default: `4.0`. Maximum semitones to shift. Positive number means shift up.

[`p`](#p){ #p }: `float` • range: [0.0, 1.0]
:   :octicons-milestone-24: Default: `0.5`. The probability of applying this transform.
