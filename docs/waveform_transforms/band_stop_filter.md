# `BandStopFilter`

_Added in v0.21.0_

Apply band-stop filtering to the input audio. Also known as notch filter or
band reject filter. It relates to the frequency mask idea in the SpecAugment paper.
This transform is similar to FrequencyMask, but has overhauled default parameters
and parameter randomization - center frequency gets picked in mel space so it is
more aligned with human hearing, which is not linear. Filter steepness
(6/12/18... dB / octave) is parametrized. Can also be set for zero-phase filtering
(will result in a 6db drop at cutoffs).
