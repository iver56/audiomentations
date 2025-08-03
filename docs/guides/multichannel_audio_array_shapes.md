# Multichannel audio array shapes

When working with audio files in Python, you may encounter two main formats for representing the data, especially when you are dealing with stereo (or multichannel) audio. These formats correspond to the shape of the numpy ndarray that holds the audio data.

## 1. Channels-first format

This format has the shape `(channels, samples)`. In the context of a stereo audio file, the number of channels would be 2 (for left and right), and samples are the individual data points in the audio file. For example, a stereo audio file with a duration of 1 second sampled at 44100 Hz would have a shape of `(2, 44100)`.

**This is the format expected by audiomentations when dealing with multichannel audio**. If you provide multichannel audio data in a different format, a `WrongMultichannelAudioShape` exception will be raised.

Note that `audiomentations` also supports mono audio, i.e. shape like `(1, samples)` or `(samples,)`

## 2. Channels-last format

This format has the shape `(samples, channels)`. Using the same stereo file example as above, the shape would be `(44100, 2)`. This format is commonly returned by the `soundfile` library when loading a stereo wav file, because channels last is the inherent data layout of a stereo wav file. This layout is the default in stereo wav files because it facilitates streaming audio, where data must be read and played back sequentially.

## Loading audio with different libraries

Different libraries in Python may return audio data in different formats. For instance, `librosa` by default returns a mono ndarray, whereas `soundfile` will return a multichannel ndarray in channels-last format when loading a stereo wav file.

Here is an example of how to load a file with each:

```python
import librosa
import soundfile as sf

# Librosa, mono
y, sr = librosa.load("stereo_audio_example.wav", sr=None, mono=True)
print(y.shape)  # (117833,)

# Librosa, multichannel
y, sr = librosa.load("stereo_audio_example.wav", sr=None, mono=False)
print(y.shape)  # (2, 117833)

# Soundfile
y, sr = sf.read("stereo_audio_example.wav")
print(y.shape)  # (117833, 2)
```

## Converting between formats

If you have audio data in the channels-last format but need it in channels-first format, you can easily convert it using the transpose operation of numpy ndarrays:

```python
import numpy as np

# Assuming y is your audio data in channels-last format
y_transposed = np.transpose(y)

# Alternative, shorter syntax:
y_transposed = y.T
```

Now, `y_transposed` will be in channels-first format and can be used with `audiomentations`.

However, there is a gotcha. Transposing the array as shown above does not change the underlying layout of the data in memory. Some audio processing libraries, especially ones that are written in C, C++ or Rust, might assume that the given NDArray is C-contiguous. If it is not C-contiguous, the code might still run without errors (depending on implementation), but your processed stereo audio might sound like it is played back at half speed, and there is a discrepancy between the left and the right channel. If you are passing audio to a function that assumes a C-contiguous data layout, you can use `np.ascontiguousarray` to make it C-contiguous:

```
y_transposed_contiguous = np.ascontiguousarray(y_transposed)
```
