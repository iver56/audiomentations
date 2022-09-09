# `ApplyImpulseResponse`

_Added in v0.7.0_

Convolve the audio with a random impulse response.
Impulse responses can be created using e.g. http://tulrich.com/recording/ir_capture/

Some datasets of impulse responses are publicly available:
- [EchoThief](http://www.echothief.com/) containing 115 impulse responses acquired in a wide range of locations.
- [The MIT McDermott](https://mcdermottlab.mit.edu/Reverb/IR_Survey.html) dataset containing 271 impulse responses acquired in everyday places.

Impulse responses are represented as wav files in the given ir_path.
