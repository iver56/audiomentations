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

# `AddGaussianNoise`

_Added in v0.1.0_

Add gaussian noise to the samples

# `AddGaussianSNR`

_Added in v0.7.0_

Add gaussian noise to the input. A random Signal to Noise Ratio (SNR) will be picked
uniformly in the decibel scale. This aligns with human hearing, which is more
logarithmic than linear.

# `ApplyImpulseResponse`

_Added in v0.7.0_

Convolve the audio with a random impulse response.
Impulse responses can be created using e.g. http://tulrich.com/recording/ir_capture/

Some datasets of impulse responses are publicly available:
- [EchoThief](http://www.echothief.com/) containing 115 impulse responses acquired in a wide range of locations.
- [The MIT McDermott](https://mcdermottlab.mit.edu/Reverb/IR_Survey.html) dataset containing 271 impulse responses acquired in everyday places.

Impulse responses are represented as wav files in the given ir_path.

# `AddShortNoises`

_Added in v0.9.0_

Mix in various (bursts of overlapping) sounds with random pauses between. Useful if your
original sound is clean and you want to simulate an environment where short noises sometimes
occur.

A folder of (noise) sounds to be mixed in must be specified.

# `AirAbsorption`

_Added in v0.25.0_

Apply a Lowpass-like filterbank with variable octave attenuation that simulates attenuation of
higher frequencies due to air absorption in some cases (10-20 degrees Celsius temperature and
30-90% humidity).

# `BandPassFilter`

_Added in v0.18.0, updated in v0.21.0_

Apply band-pass filtering to the input audio. Filter steepness (6/12/18... dB / octave)
is parametrized. Can also be set for zero-phase filtering (will result in a 6db drop at
cutoffs).

# `BandStopFilter`

_Added in v0.21.0_

Apply band-stop filtering to the input audio. Also known as notch filter or
band reject filter. It relates to the frequency mask idea in the SpecAugment paper.
This transform is similar to FrequencyMask, but has overhauled default parameters
and parameter randomization - center frequency gets picked in mel space so it is
more aligned with human hearing, which is not linear. Filter steepness
(6/12/18... dB / octave) is parametrized. Can also be set for zero-phase filtering
(will result in a 6db drop at cutoffs).

# `Clip`

_Added in v0.17.0_

Clip audio by specified values. e.g. set a_min=-1.0 and a_max=1.0 to ensure that no
samples in the audio exceed that extent. This can be relevant for avoiding integer
overflow or underflow (which results in unintended wrap distortion that can sound
horrible) when exporting to e.g. 16-bit PCM wav.

Another way of ensuring that all values stay between -1.0 and 1.0 is to apply
`PeakNormalization`.

This transform is different from `ClippingDistortion` in that it takes fixed values
for clipping instead of clipping a random percentile of the samples. Arguably, this
transform is not very useful for data augmentation. Instead, think of it as a very
cheap and harsh limiter (for samples that exceed the allotted extent) that can
sometimes be useful at the end of a data augmentation pipeline.

# `ClippingDistortion`

_Added in v0.8.0_

Distort signal by clipping a random percentage of points

The percentage of points that will be clipped is drawn from a uniform distribution between
the two input parameters min_percentile_threshold and max_percentile_threshold. If for instance
30% is drawn, the samples are clipped if they're below the 15th or above the 85th percentile.

# `FrequencyMask`

_Added in v0.7.0_

Mask some frequency band on the spectrogram.
Inspired by https://arxiv.org/pdf/1904.08779.pdf

# `Gain`

_Added in v0.11.0_

Multiply the audio by a random amplitude factor to reduce or increase the volume. This
technique can help a model become somewhat invariant to the overall gain of the input audio.

Warning: This transform can return samples outside the [-1, 1] range, which may lead to
clipping or wrap distortion, depending on what you do with the audio in a later stage.
See also https://en.wikipedia.org/wiki/Clipping_(audio)#Digital_clipping

# `GainTransition`

_Added in v0.22.0_

Gradually change the volume up or down over a random time span. Also known as
fade in and fade out. The fade works on a logarithmic scale, which is natural to
human hearing.

The way this works is that it picks two gains: a first gain and a second gain.
Then it picks a time range for the transition between those two gains.
Note that this transition can start before the audio starts and/or end after the
audio ends, so the output audio can start or end in the middle of a transition.
The gain starts at the first gain and is held constant until the transition start.
Then it transitions to the second gain. Then that gain is held constant until the
end of the sound.

# `HighPassFilter`

_Added in v0.18.0, updated in v0.21.0_

Apply high-pass filtering to the input audio of parametrized filter steepness (6/12/18... dB / octave).
Can also be set for zero-phase filtering (will result in a 6db drop at cutoff).

# `HighShelfFilter`

_Added in v0.21.0_

A high shelf filter is a filter that either boosts (increases amplitude) or cuts
(decreases amplitude) frequencies above a certain center frequency. This transform
applies a high-shelf filter at a specific center frequency in hertz.
The gain at nyquist frequency is controlled by `{min,max}_gain_db` (note: can be positive or negative!).
Filter coefficients are taken from [the W3 Audio EQ Cookbook](https://www.w3.org/TR/audio-eq-cookbook/)

# `LowPassFilter`

_Added in v0.18.0, updated in v0.21.0_

Apply low-pass filtering to the input audio of parametrized filter steepness (6/12/18... dB / octave).
Can also be set for zero-phase filtering (will result in a 6db drop at cutoff).

# `LowShelfFilter`

_Added in v0.21.0_

A low shelf filter is a filter that either boosts (increases amplitude) or cuts
(decreases amplitude) frequencies below a certain center frequency. This transform
applies a low-shelf filter at a specific center frequency in hertz.
The gain at DC frequency is controlled by `{min,max}_gain_db` (note: can be positive or negative!).
Filter coefficients are taken from [the W3 Audio EQ Cookbook](https://www.w3.org/TR/audio-eq-cookbook/)

# `Mp3Compression`

_Added in v0.12.0_

Compress the audio using an MP3 encoder to lower the audio quality. This may help machine
learning models deal with compressed, low-quality audio.

This transform depends on either lameenc or pydub/ffmpeg.

Note that bitrates below 32 kbps are only supported for low sample rates (up to 24000 hz).

Note: When using the lameenc backend, the output may be slightly longer than the input due
to the fact that the LAME encoder inserts some silence at the beginning of the audio.

# `Lambda`

_Added in v0.26.0_

Apply a user-defined transform (callable) to the signal.

# `Limiter`

_Added in v0.26.0_

A simple audio limiter (dynamic range compression).
Note: This transform also delays the signal by a fraction of the attack time.

# `LoudnessNormalization`

_Added in v0.14.0_

Apply a constant amount of gain to match a specific loudness. This is an implementation of
ITU-R BS.1770-4.

Warning: This transform can return samples outside the [-1, 1] range, which may lead to
clipping or wrap distortion, depending on what you do with the audio in a later stage.
See also https://en.wikipedia.org/wiki/Clipping_(audio)#Digital_clipping

# `Normalize`

_Added in v0.6.0_

Apply a constant amount of gain, so that highest signal level present in the sound becomes
0 dBFS, i.e. the loudest level allowed if all samples must be between -1 and 1. Also known
as peak normalization.

# `Padding`

_Added in v0.23.0_

Apply padding to the audio signal - take a fraction of the end or the start of the
audio and replace that part with padding. This can be useful for preparing ML models
with constant input length for padded inputs.

# `PeakingFilter`

_Added in v0.21.0_

Add a biquad peaking filter transform

# `PitchShift`

_Added in v0.4.0_

Pitch shift the sound up or down without changing the tempo

# `PolarityInversion`

_Added in v0.11.0_

Flip the audio samples upside-down, reversing their polarity. In other words, multiply the
waveform by -1, so negative values become positive, and vice versa. The result will sound
the same compared to the original when played back in isolation. However, when mixed with
other audio sources, the result may be different. This waveform inversion technique
is sometimes used for audio cancellation or obtaining the difference between two waveforms.
However, in the context of audio data augmentation, this transform can be useful when
training phase-aware machine learning models.

# `Resample`

_Added in v0.8.0_

Resample signal using librosa.core.resample

To do downsampling only set both minimum and maximum sampling rate lower than original
sampling rate and vice versa to do upsampling only.

# `Reverse`

_Added in v0.18.0_

Reverse the audio. Also known as time inversion. Inversion of an audio track along its time
axis relates to the random flip of an image, which is an augmentation technique that is
widely used in the visual domain. This can be relevant in the context of audio
classification. It was successfully applied in the paper
[AudioCLIP: Extending CLIP to Image, Text and Audio](https://arxiv.org/pdf/2106.13043.pdf).

# `RoomSimulator`

_Added in v0.23.0_

A ShoeBox Room Simulator. Simulates a cuboid of parametrized size and average surface absorption coefficient. It also includes a source
and microphones in parametrized locations.

Use it when you want a ton of synthetic room impulse responses of specific configurations
characteristics or simply to quickly add reverb for augmentation purposes

# `SevenBandParametricEQ`

_Added in v0.24.0_

Adjust the volume of different frequency bands. This transform is a 7-band
parametric equalizer - a combination of one low shelf filter, five peaking filters
and one high shelf filter, all with randomized gains, Q values and center frequencies.

Because this transform changes the timbre, but keeps the overall "class" of the
sound the same (depending on application), it can be used for data augmentation to
make ML models more robust to various frequency spectrums. Many things can affect
the spectrum, like room acoustics, any objects between the microphone and
the sound source, microphone type/model and the distance between the sound source
and the microphone.

The seven bands have center frequencies picked in the following ranges (min-max):
42-95 hz
91-204 hz
196-441 hz
421-948 hz
909-2045 hz
1957-4404 hz
4216-9486 hz

# `Shift`

_Added in v0.5.0_

Shift the samples forwards or backwards, with or without rollover

# `TanhDistortion`

_Added in v0.19.0_

Apply tanh (hyperbolic tangent) distortion to the audio. This technique is sometimes
used for adding distortion to guitar recordings. The tanh() function can give a rounded
"soft clipping" kind of distortion, and the distortion amount is proportional to the
loudness of the input and the pre-gain. Tanh is symmetric, so the positive and
negative parts of the signal are squashed in the same way. This transform can be
useful as data augmentation because it adds harmonics. In other words, it changes
the timbre of the sound.

See this page for examples: [http://gdsp.hf.ntnu.no/lessons/3/17/](http://gdsp.hf.ntnu.no/lessons/3/17/)

# `TimeMask`

_Added in v0.7.0_

Make a randomly chosen part of the audio silent.
Inspired by https://arxiv.org/pdf/1904.08779.pdf

# `TimeStretch`

_Added in v0.2.0_

Time stretch the signal without changing the pitch

# `Trim`

_Added in v0.7.0_

Trim leading and trailing silence from an audio signal using `librosa.effects.trim`
