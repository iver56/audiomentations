# `GainCompensation`

_Added in v0.33.0_

Gain up or down the audio after the given transform (or set of transforms) has
processed the audio. `GainCompensation` can be useful for compensating for any gain
differences introduced by a (set of) transform(s), like `ApplyImpulseResponse`,
`ApplyBackgroundNoise`, `Clip` and many others. `GainCompensation` ensures that the
processed audio's RMS (Root Mean Square) or LUFS (Loudness Units Full Scale) matches
the original.

# PostGain API

[`transform`](#transform){ #transform }: `Callable[[np.ndarray, int], np.ndarray]`
:   :octicons-milestone-24: A callable to be applied. It should input
    samples (ndarray), sample_rate (int) and optionally some user-defined
    keyword arguments.

[`method`](#method){ #method }: `str` • choices: `"same_rms"`, `"same_lufs"` or `"peak_normalize_always"`
:   :octicons-milestone-24: This parameter defines the method for choosing the post gain amount.

    * `"same_rms"`: The sound gets post-gained so that the RMS (Root Mean Square) of
        the output matches the RMS of the input.
    * `"same_lufs"`: The sound gets post-gained so that the LUFS (Loudness Units Full Scale) of
        the output matches the LUFS of the input.
    * `"peak_normalize_always"`: The sound gets peak normalized (gained up or down so
        that the absolute value of the most extreme sample in the output is 1.0)
    * `"peak_normalize_if_too_loud"`: The sound gets peak normalized if it is too
        loud (max absolute value greater than 1.0). This option can be useful for
        avoiding clipping.
