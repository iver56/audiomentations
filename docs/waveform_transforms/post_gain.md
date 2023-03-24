# `PostGain`

_Added in v0.30.0_

Gain up or down the audio after the given transform (or set of transforms) has
processed the audio. There are several methods that determine how the audio should
be gained. `PostGain` can be useful for compensating for any gain differences introduced
by a (set of) transform(s), or for preventing clipping in the output.

# PostGain API

[`transform`](#transform){ #transform }: `Callable[[np.ndarray, int], np.ndarray]`
:   :octicons-milestone-24: A callable to be applied. It should input
    samples (ndarray), sample_rate (int) and optionally some user-defined
    keyword arguments.

[`method`](#method){ #method }: `str` • choices: `"same_rms"`, `"same_lufs"` or `"peak_normalize_always"`
:   :octicons-milestone-24: This parameter defines the method for choosing the post gain amount.

    * `"same_rms"`: The sound will be post-gained so that the RMS (Root Mean Square) of
        the output matches the RMS of the input.
    * `"same_lufs"`: The sound will be post-gained so that the LUFS (Loudness Units Full Scale) of
        the output matches the LUFS of the input.
    * `"peak_normalize_always"`: The sound will be gained so that the absolute value of
        the most extreme sample in the output will be 1.0
