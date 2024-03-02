# `Mp3Compression`

_Added in v0.12.0_

Compress the audio using an MP3 encoder to lower the audio quality. This may help machine
learning models deal with compressed, low-quality audio.

This transform depends on either lameenc or pydub/ffmpeg.

Note that bitrates below 32 kbps are only supported for low sample rates (up to 24000 Hz).

Note: When using the `"lameenc"` backend, the output may be slightly longer than the input due
to the fact that the LAME encoder inserts some silence at the beginning of the audio.

Warning: This transform writes to disk, so it may be slow.

# Mp3Compression API

[`min_bitrate`](#min_bitrate){ #min_bitrate }: `int` • unit: kbps • range: [8, `max_bitrate`]
:   :octicons-milestone-24: Default: `8`. Minimum bitrate in kbps

[`max_bitrate`](#max_bitrate){ #max_bitrate }: `int` • unit: kbps • range: [`min_bitrate`, 320]
:   :octicons-milestone-24: Default: `64`. Maximum bitrate in kbps

[`backend`](#backend){ #backend }: `str` • choices: `"pydub"`, `"lameenc"`
:   :octicons-milestone-24: Default: `"pydub"`.

    * `"pydub"`: May use ffmpeg under the hood. Pro: Seems to avoid introducing latency in
        the output. Con: Slightly slower than `"lameenc"`.
    * `"lameenc"`: Pro: With this backend you can set the quality parameter in addition
        to the bitrate (although this parameter is not exposed in the audiomentations API
        yet). Con: Seems to introduce some silence at the start of the audio.

[`p`](#p){ #p }: `float` • range: [0.0, 1.0]
:   :octicons-milestone-24: Default: `0.5`. The probability of applying this transform.
