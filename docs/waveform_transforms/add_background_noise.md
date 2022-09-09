# `AddBackgroundNoise`

_Added in v0.9.0_

Mix in another sound, e.g. a background noise. Useful if your original sound is clean and
you want to simulate an environment where background noise is present.

Can also be used for mixup, as in https://arxiv.org/pdf/1710.09412.pdf

A folder of (background noise) sounds to be mixed in must be specified. These sounds should
ideally be at least as long as the input sounds to be transformed. Otherwise, the background
sound will be repeated, which may sound unnatural.

Note that the gain of the added noise is relative to the amount of signal in the input. This
implies that if the input is completely silent, no noise will be added.

Here are some examples of datasets that can be downloaded and used as background noise:

* https://github.com/karolpiczak/ESC-50#download
* https://github.com/microsoft/DNS-Challenge/
