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
