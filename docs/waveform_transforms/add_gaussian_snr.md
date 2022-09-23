# `AddGaussianSNR`

_Added in v0.7.0_

Add gaussian noise to the input. A random Signal to Noise Ratio (SNR) will be picked
uniformly in the decibel scale. This aligns with human hearing, which is more
logarithmic than linear.


## Usage example

```python
from audiomentations import AddGaussianSNR

transform = AddGaussianSNR(
    min_snr_in_db=5.0,
    max_snr_in_db=40.0,
    p=1.0
)

augmented_sound = transform(my_waveform_ndarray, sample_rate=16000)
```

## AddGaussianSNR API

[`min_snr_in_db`](#min_snr_in_db){ #min_snr_in_db }: `float` (unit: Decibel)
:   :octicons-milestone-24: Default: `5.0`. Minimum signal-to-noise ratio in dB. A lower
    number means more noise.

[`max_snr_in_db`](#max_snr_in_db){ #max_snr_in_db }: `float` (unit: decibel)
:   :octicons-milestone-24: Default: `40.0`. Maximum signal-to-noise ratio in dB. A
    greater number means less noise.

[`p`](#p){ #p }: `float`
:   :octicons-milestone-24: Default: `0.5`. The probability of applying this transform.
