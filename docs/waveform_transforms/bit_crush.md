# `BitCrush`

_Added in v0.35.0_

Apply a bit crush effect to the audio by reducing the bit depth. In other words, it
reduces the number of bits that can be used for representing each audio sample.
This adds quantization noise, and affects dynamic range. This transform does not apply
dithering.

For more information, see

* [Resolution reduction :octicons-link-external-16:](https://en.wikipedia.org/wiki/Bitcrusher#Resolution_reduction){target=_blank} on Wikipedia
* [Intro to bit reduction :octicons-link-external-16:](http://gdsp.hf.ntnu.no/lessons/1/4/){target=_blank} by NTNU, Department of Music, Music Technology

## Input-output example

Here we reduce the bit depth from 16 to 6 bits per sample

![Input-output waveforms and spectrograms](BitCrush.webp)

| Input sound                                                                     | Transformed sound                                                                     |
|---------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| <audio controls><source src="../BitCrush_input.flac" type="audio/flac"></audio> | <audio controls><source src="../BitCrush_transformed.flac" type="audio/flac"></audio> | 

## Usage example

```python
from audiomentations import BitCrush

transform = BitCrush(min_bit_depth=5, max_bit_depth=14, p=1.0)

augmented_sound = transform(my_waveform_ndarray, sample_rate=16000)
```

# BitCrush API

[`min_bit_depth`](#min_bit_depth){ #min_bit_depth }: `int` • unit: bits • range: [1, 32]
:   :octicons-milestone-24: Minimum bit depth the audio will be "converted" to

[`max_bit_depth`](#max_bit_depth){ #max_bit_depth }: `int` • unit: bits • range: [1, 32]
:   :octicons-milestone-24: Maximum bit depth the audio will be "converted" to

[`p`](#p){ #p }: `float` • range: [0.0, 1.0]
:   :octicons-milestone-24: Default: `0.5`. The probability of applying this transform.

## Source code :octicons-mark-github-16:

[audiomentations/augmentations/bit_crush.py :octicons-link-external-16:](https://github.com/iver56/audiomentations/blob/main/audiomentations/augmentations/bit_crush.py){target=_blank}
