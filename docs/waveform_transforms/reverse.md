# `Reverse`

_Added in v0.18.0_

Reverse the audio. Also known as time inversion. Inversion of an audio track along its time
axis relates to the random flip of an image, which is an augmentation technique that is
widely used in the visual domain. This can be relevant in the context of audio
classification. It was successfully applied in the paper
[AudioCLIP: Extending CLIP to Image, Text and Audio](https://arxiv.org/pdf/2106.13043.pdf).

# Reverse API

[`p`](#p){ #p }: `float` • range: [0.0, 1.0]
:   :octicons-milestone-24: Default: `0.5`. The probability of applying this transform.
