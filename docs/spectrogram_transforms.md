audiomentations is in a very early (read: not very useful yet) stage when it comes to spectrogram transforms. Consider applying [waveform transforms](waveform_transforms.md) before converting your waveforms to spectrograms, or check out [alternative libraries](alternatives.md)  

# `SpecChannelShuffle`

_Added in v0.13.0_

Shuffle the channels of a multichannel spectrogram. This can help combat positional bias.

# `SpecFrequencyMask`

_Added in v0.13.0_

Mask a set of frequencies in a spectrogram, à la Google AI SpecAugment. This type of data
augmentation has proved to make speech recognition models more robust.

The masked frequencies can be replaced with either the mean of the original values or a
given constant (e.g. zero).
