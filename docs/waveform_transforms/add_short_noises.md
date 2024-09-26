# `AddShortNoises`

_Added in v0.9.0_

Mix in various (bursts of overlapping) sounds with random pauses between. Useful if your
original sound is clean and you want to simulate an environment where short noises sometimes
occur.

A folder of (noise) sounds to be mixed in must be specified.

## Input-output example

Here we add some short noise sounds to a voice recording.

![Input-output waveforms and spectrograms](AddShortNoises.webp)

| Input sound                                                                           | Transformed sound                                                                           |
|---------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------|
| <audio controls><source src="../AddShortNoises_input.flac" type="audio/flac"></audio> | <audio controls><source src="../AddShortNoises_transformed.flac" type="audio/flac"></audio> | 


## Usage examples


=== "Noise RMS relative to whole input"

    ```python
    from audiomentations import AddShortNoises, PolarityInversion

    transform = AddShortNoises(
        sounds_path="/path/to/folder_with_sound_files",
        min_snr_db=3.0,
        max_snr_db=30.0,
        noise_rms="relative_to_whole_input",
        min_time_between_sounds=2.0,
        max_time_between_sounds=8.0,
        noise_transform=PolarityInversion(),
        p=1.0
    )

    augmented_sound = transform(my_waveform_ndarray, sample_rate=16000)
    ```

=== "Absolute RMS"

    ```python
    from audiomentations import AddShortNoises, PolarityInversion

    transform = AddShortNoises(
        sounds_path="/path/to/folder_with_sound_files",
        min_absolute_noise_rms_db=-50.0,
        max_absolute_noise_rms_db=-20.0,        
        noise_rms="absolute",
        min_time_between_sounds=2.0,
        max_time_between_sounds=8.0,
        noise_transform=PolarityInversion(),
        p=1.0
    )

    augmented_sound = transform(my_waveform_ndarray, sample_rate=16000)
    ```

## AddShortNoises API

[`sounds_path`](#sounds_path){ #sounds_path }: `Union[List[Path], List[str], Path, str]`
:   :octicons-milestone-24: A path or list of paths to audio file(s) and/or folder(s)
    with audio files. Can be str or Path instance(s). The audio files given here are
    supposed to be (short) noises.

~~[`min_snr_in_db`](#min_snr_in_db){ #min_snr_in_db }: `float` • unit: Decibel~~
:   :warning: Deprecated as of v0.31.0, removed as of v0.38.0. Use [`min_snr_db`](#min_snr_db) instead

~~[`max_snr_in_db`](#max_snr_in_db){ #max_snr_in_db }: `float` • unit: Decibel~~
:   :warning: Deprecated as of v0.31.0, removed as of v0.38.0. Use [`max_snr_db`](#max_snr_db) instead

[`min_snr_db`](#min_snr_db){ #min_snr_db }: `float` • unit: Decibel
:   :octicons-milestone-24: Default: `-6.0`. Minimum signal-to-noise ratio in dB. A lower
    value means the added sounds/noises will be louder. This gets ignored if `noise_rms`
    is set to `"absolute"`.

[`max_snr_db`](#max_snr_db){ #max_snr_db }: `float` • unit: Decibel
:   :octicons-milestone-24: Default: `18.0`. Maximum signal-to-noise ratio in dB. A
    lower value means the added sounds/noises will be louder. This gets ignored if
    `noise_rms` is set to `"absolute"`.

[`min_time_between_sounds`](#min_time_between_sounds){ #min_time_between_sounds }: `float` • unit: seconds
:   :octicons-milestone-24: Default: `2.0`. Minimum pause time (in seconds) between the
    added sounds/noises

[`max_time_between_sounds`](#max_time_between_sounds){ #max_time_between_sounds }: `float` • unit: seconds
:   :octicons-milestone-24: Default: `8.0`. Maximum pause time (in seconds) between the
    added sounds/noises

[`noise_rms`](#noise_rms){ #noise_rms }: `str` • choices: `"absolute"`, `"relative"`, `"relative_to_whole_input"`
:   :octicons-milestone-24: Default: `"relative"` (<=v0.27), but will be changed to
    `"relative_to_whole_input"` in a future version.

    This parameter defines how the noises will be added to the audio input.

    * `"relative"`: the RMS value of the added noise will be proportional to the RMS value of
        the input sound calculated only for the region where the noise is added.
    * `"absolute"`: the added noises will have an RMS independent of the RMS of the input audio
        file.
    * `"relative_to_whole_input"`: the RMS of the added noises will be
        proportional to the RMS of the whole input sound.

[`min_absolute_noise_rms_db`](#min_absolute_noise_rms_db){ #min_absolute_noise_rms_db }: `float` • unit: Decibel
:   :octicons-milestone-24: Default: `-50.0`. Is only used if `noise_rms` is set to
    `"absolute"`. It is the minimum RMS value in dB that the added noise can take. The
    lower the RMS is, the lower will the added sound be.

[`max_absolute_noise_rms_db`](#max_absolute_noise_rms_db){ #max_absolute_noise_rms_db }: `float` • unit: seconds
:   :octicons-milestone-24: Default: `-20.0`. Is only used if `noise_rms` is set to
    `"absolute"`. It is the maximum RMS value in dB that the added noise can take. Note
    that this value can not exceed 0.

[`add_all_noises_with_same_level`](#add_all_noises_with_same_level){ #add_all_noises_with_same_level }: `bool`
:   :octicons-milestone-24: Default: `False`. Whether to add all the short noises
    (within one audio snippet) with the same SNR. If `noise_rms` is set to `"absolute"`,
    the RMS is used instead of SNR. The target SNR (or RMS) will change every time the
    parameters of the transform are randomized.

[`include_silence_in_noise_rms_estimation`](#include_silence_in_noise_rms_estimation){ #include_silence_in_noise_rms_estimation }: `bool`
:   :octicons-milestone-24: Default: `True`. It chooses how the RMS of
    the noises to be added will be calculated. If this option is set to False, the silence
    in the noise files will be disregarded in the RMS calculation. It is useful for
    non-stationary noises where silent periods occur.

[`burst_probability`](#burst_probability){ #burst_probability }: `float`
:   :octicons-milestone-24: Default: `0.22`. For every noise that gets added, there
    is a probability of adding an extra burst noise that overlaps with the noise. This
    parameter controls that probability. `min_pause_factor_during_burst` and
    `max_pause_factor_during_burst` control the amount of overlap.

[`min_pause_factor_during_burst`](#min_pause_factor_during_burst){ #min_pause_factor_during_burst }: `float`
:   :octicons-milestone-24: Default: `0.1`. Min value of how far into the current sound (as
    fraction) the burst sound should start playing. The value must be greater than 0.

[`max_pause_factor_during_burst`](#max_pause_factor_during_burst){ #max_pause_factor_during_burst }: `float`
:   :octicons-milestone-24: Default: `1.1`. Max value of how far into the current sound (as
    fraction) the burst sound should start playing. The value must be greater than 0.

[`min_fade_in_time`](#min_fade_in_time){ #min_fade_in_time }: `float` • unit: seconds
:   :octicons-milestone-24: Default: `0.005`. Min noise fade in time in seconds. Use a
    value larger than 0 to avoid a "click" at the start of the noise.

[`max_fade_in_time`](#max_fade_in_time){ #max_fade_in_time }: `float` • unit: seconds
:   :octicons-milestone-24: Default: `0.08`. Max noise fade in time in seconds. Use a
    value larger than 0 to avoid a "click" at the start of the noise.

[`min_fade_out_time`](#min_fade_out_time){ #min_fade_out_time }: `float` • unit: seconds
:   :octicons-milestone-24: Default: `0.01`. Min sound/noise fade out time in seconds.
    Use a value larger than 0 to avoid a "click" at the end of the sound/noise.

[`max_fade_out_time`](#max_fade_out_time){ #max_fade_out_time }: `float` • unit: seconds
:   :octicons-milestone-24: Default: `0.1`. Max sound/noise fade out time in seconds.
    Use a value larger than 0 to avoid a "click" at the end of the sound/noise.

[`signal_gain_in_db_during_noise`](#signal_gain_in_db_during_noise){ #signal_gain_in_db_during_noise }: `float` • unit: Decibel
:   :warning: Deprecated as of v0.31.0. Use [`signal_gain_db_during_noise`](#signal_gain_db_during_noise) instead

[`signal_gain_db_during_noise`](#signal_gain_db_during_noise){ #signal_gain_in_db_during_noise }: `float` • unit: Decibel
:   :octicons-milestone-24: Default: `0.0`. Gain applied to the signal during a short noise.
    When fading the signal to the custom gain, the same fade times are used as
    for the noise, so it's essentially cross-fading. The default value (0.0) means
    the signal will not be gained. If set to a very low value, e.g. -100.0, this
    feature could be used for completely replacing the signal with the noise.
    This could be relevant in some use cases, for example:

    * replace the signal with another signal of a similar class (e.g. replace some
        speech with a cough)
    * simulate an ECG off-lead condition (electrodes are temporarily disconnected)

[`noise_transform`](#noise_transform){ #noise_transform }: `Optional[Callable[[NDArray[np.float32], int], NDArray[np.float32]]]`
:   :octicons-milestone-24: Default: `None`. A callable waveform transform (or
    composition of transforms) that gets applied to noises before they get mixed in.

[`p`](#p){ #p }: `float` • range: [0.0, 1.0]
:   :octicons-milestone-24: Default: `0.5`. The probability of applying this transform.

[`lru_cache_size`](#lru_cache_size){ #lru_cache_size }: `int`
:   :octicons-milestone-24: Default: `64`. Maximum size of the LRU cache for storing
    noise files in memory

## Source code :octicons-mark-github-16:

[audiomentations/augmentations/add_short_noises.py :octicons-link-external-16:](https://github.com/iver56/audiomentations/blob/main/audiomentations/augmentations/add_short_noises.py){target=_blank}
