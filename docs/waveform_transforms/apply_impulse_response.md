# `ApplyImpulseResponse`

_Added in v0.7.0_

Convolve the audio with a randomly selected impulse response.
Impulse responses can be created using e.g. [http://tulrich.com/recording/ir_capture/](http://tulrich.com/recording/ir_capture/)

Some datasets of impulse responses are publicly available:
- [EchoThief](http://www.echothief.com/) containing 115 impulse responses acquired in a
 wide range of locations.
- [The MIT McDermott](https://mcdermottlab.mit.edu/Reverb/IR_Survey.html) dataset
 containing 271 impulse responses acquired in everyday places.

Impulse responses are represented as audio (ideally wav) files in the given `ir_path`.

## ApplyImpulseResponse API

[`ir_path`](#ir_path){ #ir_path }: `Union[List[Path], List[str], str, Path]`
:   :octicons-milestone-24: A path or list of paths to audio file(s) and/or folder(s) with
    audio files. Can be `str` or `Path` instance(s). The audio files given here are
    supposed to be impulse responses.

[`p`](#p){ #p }: `float` (range: [0.0, 1.0])
:   :octicons-milestone-24: Default: `0.5`. The probability of applying this transform.

[`lru_cache_size`](#lru_cache_size){ #lru_cache_size }: `int`
:   :octicons-milestone-24: Default: `128`. Maximum size of the LRU cache for storing
    impulse response files in memory.

[`leave_length_unchanged`](#leave_length_unchanged){ #leave_length_unchanged }: `bool`
:   :octicons-milestone-24: Default: `True`. When set to `True`, the tail of the sound
    (e.g. reverb at the end) will be chopped off so that the length of the output is
    equal to the length of the input.
