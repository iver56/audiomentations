# `LowShelfFilter`

_Added in v0.21.0_

A low shelf filter is a filter that either boosts (increases amplitude) or cuts
(decreases amplitude) frequencies below a certain center frequency. This transform
applies a low-shelf filter at a specific center frequency in hertz.
The gain at DC frequency is controlled by `{min,max}_gain_db` (note: can be positive or negative!).
Filter coefficients are taken from [the W3 Audio EQ Cookbook](https://www.w3.org/TR/audio-eq-cookbook/)
