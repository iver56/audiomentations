# `Limiter`

_Added in v0.26.0_

A simple audio limiter (dynamic range compression).
Note: This transform also delays the signal by a fraction of the attack time.

* The _threshold_ determines the audio level above which the limiter kicks in.
* The _attack_ time is how quickly the limiter kicks in once the audio signal starts exceeding the threshold.
* The _release_ time determines how quickly the limiter stops working after the signal drops below the threshold.


# Limiter API

[`min_threshold_db`](#min_threshold_db){ #min_threshold_db }: `float` • unit: Decibel
:   :octicons-milestone-24: Default: `-24.0`. Minimum threshold

[`max_threshold_db`](#max_threshold_db){ #max_threshold_db }: `float` • unit: Decibel
:   :octicons-milestone-24: Default: `-2.0`. Maximum threshold

[`min_attack`](#min_attack){ #min_attack }: `float` • unit: seconds
:   :octicons-milestone-24: Default: `0.0005`. Minimum attack time

[`max_attack`](#max_attack){ #max_attack }: `float` • unit: seconds
:   :octicons-milestone-24: Default: `0.025`. Maximum attack time

[`min_release`](#min_release){ #min_release }: `float` • unit: seconds
:   :octicons-milestone-24: Default: `0.05`. Minimum release time

[`max_release`](#max_release){ #max_release }: `float` • unit: seconds
:   :octicons-milestone-24: Default: `0.7`. Maximum release time

[`threshold_mode`](#threshold_mode){ #threshold_mode }: `str` • choices: `"relative_to_signal_peak"`, `"absolute"`
:   :octicons-milestone-24: Default: `relative_to_signal_peak`.

    * `"relative_to_signal_peak"` means the threshold is relative to peak of the signal.
    * `"absolute"` means the threshold is relative to 0 dBFS, so it doesn't depend
     on the peak of the signal.

[`p`](#p){ #p }: `float` • range: [0.0, 1.0]
:   :octicons-milestone-24: Default: `0.5`. The probability of applying this transform.
