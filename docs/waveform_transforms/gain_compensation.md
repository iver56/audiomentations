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

[`loudness_metric`](#loudness_metric){ #loudness_metric }: `str` • choices: `"rms"`, `"lufs"`
:   :octicons-milestone-24: Default: `"rms"`. This parameter defines the loudness metric
    used when calculating the gain amount.

    * `"rms"`: The sound gets post-gained so that the RMS (Root Mean Square) of
        the output matches the RMS of the input.
    * `"lufs"`: The sound gets post-gained so that the LUFS (Loudness Units Full Scale) of
        the output matches the LUFS of the input. This is slower than the "rms", but is
        more aligned with human's perception of loudness
