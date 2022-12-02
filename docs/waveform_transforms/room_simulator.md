# `RoomSimulator`

_Added in v0.23.0_

A ShoeBox Room Simulator. Simulates a cuboid of parametrized size and average surface absorption coefficient. It also includes a source
and microphones in parametrized locations.

Use it when you want a ton of synthetic room impulse responses of specific configurations
characteristics or simply to quickly add reverb for augmentation purposes

## RoomSimulator API

[`min_size_x`](#min_size_x){ #min_size_x }: `float` • unit: meters
:   :octicons-milestone-24: Default: `3.6`. Minimum width (x coordinate) of the room in meters

[`max_size_x`](#max_size_x){ #max_size_x }: `float` • unit: meters
:   :octicons-milestone-24: Default: `5.6`. Maximum width of the room in meters

[`min_size_y`](#min_size_y){ #min_size_y }: `float` • unit: meters
:   :octicons-milestone-24: Default: `3.6`. Minimum depth (y coordinate) of the room in meters

[`max_size_y`](#max_size_y){ #max_size_y }: `float` • unit: meters
:   :octicons-milestone-24: Default: `3.9`. Maximum depth of the room in meters

[`min_size_z`](#min_size_z){ #min_size_z }: `float` • unit: meters
:   :octicons-milestone-24: Default: `2.4`. Minimum height (z coordinate) of the room in meters

[`max_size_z`](#max_size_z){ #max_size_z }: `float` • unit: meters
:   :octicons-milestone-24: Default: `3.0`. Maximum height of the room in meters

[`min_absorption_value`](#min_absorption_value){ #min_absorption_value }: `float`
:   :octicons-milestone-24: Default: `0.075`. Minimum absorption coefficient value.
    When [`calculation_mode`](#calculation_mode){ #calculation_mode } is `"absorption"`
    it will set the given coefficient value for the surfaces of the room (walls,
    ceilings, and floor). This coefficient takes values between 0 (fully reflective
    surface) and 1 (fully absorbing surface).
    
    Example values (may differ!):
    
    | Environment                 | Coefficient value |
    | --------------------------- | ----------------- |
    | Studio with acoustic panels | > 0.40            |
    | Office / Library            | ~ 0.15            |
    | Factory                     | ~ 0.05            |

[`max_absorption_value`](#max_absorption_value){ #max_absorption_value }: `float`
:   :octicons-milestone-24: Default: `0.4`. Maximum absorption coefficient value. See
    [`min_absorption_value`](#min_absorption_value){ #min_absorption_value } for more
    info.

[`min_target_rt60`](#min_target_rt60){ #min_target_rt60 }: `float` • unit: seconds
:   :octicons-milestone-24: Default: `0.15`. Minimum target RT60. RT60 is defined as the
    measure of the time after the sound source ceases that it takes for the sound
    pressure level to reduce by 60 dB. When
    [`calculation_mode`](#calculation_mode){ #calculation_mode } is `"rt60"`, it tries
    to set the absorption value of the surfaces of the room to achieve a target RT60
    (in seconds). Note that this parameter changes only the materials (absorption
    coefficients) of the surfaces, _not_ the dimension of the rooms.

    Example values (may differ!):
    
    | Environment      | RT60  |
    | ---------------- | ----- |
    | Recording studio | 0.3 s |
    | Office           | 0.5 s |
    | Concert hall     | 1.5 s |

[`max_target_rt60`](#max_target_rt60){ #max_target_rt60 }: `float` • unit: seconds
:   :octicons-milestone-24: Default: `0.8`. Maximum target RT60. See
    [`min_target_rt60`](#min_target_rt60){ #min_target_rt60 } for more info.

[`min_source_x`](#min_source_x){ #min_source_x }: `float` • unit: meters
:   :octicons-milestone-24: Default: `0.1`. Minimum x location of the source

[`max_source_x`](#max_source_x){ #max_source_x }: `float` • unit: meters
:   :octicons-milestone-24: Default: `3.5`. Maximum x location of the source

[`min_source_y`](#min_source_y){ #min_source_y }: `float` • unit: meters
:   :octicons-milestone-24: Default: `0.1`. Minimum y location of the source

[`max_source_x`](#max_source_x){ #max_source_x }: `float` • unit: meters
:   :octicons-milestone-24: Default: `2.7`. Maximum y location of the source

[`min_source_z`](#min_source_z){ #min_source_z }: `float` • unit: meters
:   :octicons-milestone-24: Default: `1.0`. Minimum z location of the source

[`max_source_x`](#max_source_x){ #max_source_x }: `float` • unit: meters
:   :octicons-milestone-24: Default: `2.1`. Maximum z location of the source

[`min_mic_distance`](#min_mic_distance){ #min_mic_distance }: `float` • unit: meters
:   :octicons-milestone-24: Default: `0.15`. Minimum distance of the microphone from the
    source in meters

[`max_mic_distance`](#max_mic_distance){ #max_mic_distance }: `float` • unit: meters
:   :octicons-milestone-24: Default: `0.35`. Maximum distance of the microphone from the
    source in meters

[`min_mic_azimuth`](#min_mic_azimuth){ #min_mic_azimuth }: `float` • unit: radians
:   :octicons-milestone-24: Default: `-π`. Minimum azimuth (angle around z axis) of the
    microphone relative to the source.

[`max_mic_azimuth`](#max_mic_azimuth){ #max_mic_azimuth }: `float` • unit: radians
:   :octicons-milestone-24: Default: `π`. Maximum azimuth (angle around z axis) of the
    microphone relative to the source.

[`min_mic_elevation`](#min_mic_elevation){ #min_mic_elevation }: `float` • unit: radians
:   :octicons-milestone-24: Default: `-π`. Minimum elevation of the microphone relative
    to the source, in radians.

[`max_mic_elevation`](#max_mic_elevation){ #max_mic_elevation }: `float` • unit: radians
:   :octicons-milestone-24: Default: `π`. Maximum elevation of the microphone relative
    to the source, in radians.

[`calculation_mode`](#calculation_mode){ #calculation_mode }: `str` • choices: `"rt60"`, `"absorption"`
:   :octicons-milestone-24: Default: `"absorption"`. When set to `"absorption"`, it will
    create the room with surfaces based on
    [`min_absorption_value`](#min_absorption_value){ #min_absorption_value } and
    [`max_absorption_value`](#max_absorption_value){ #max_absorption_value }. If set to
    `"rt60"` it will try to assign surface materials that lead to a room impulse
    response with target rt60 given by
    [`min_target_rt60`](#min_target_rt60){ #min_target_rt60 } and
    [`max_target_rt60`](#max_target_rt60){ #max_target_rt60 }

[`use_ray_tracing`](#use_ray_tracing){ #use_ray_tracing }: `bool`
:   :octicons-milestone-24: Default: `True`. Whether to use ray_tracing or not (slower
    but much more accurate). Disable this if you need speed but do not really care for
    incorrect results.

[`max_order`](#max_order){ #max_order }: `Optional[int]` • range: [1, 30]
:   :octicons-milestone-24: Default: `None`. Maximum order of reflections for the Image
    Source Model. E.g. a value of 1 will only add first order reflections while a value
    of 30 will add a diffuse reverberation tail.

[`leave_length_unchanged`](#leave_length_unchanged){ #leave_length_unchanged }: `bool`
:   :octicons-milestone-24: Default: `False`. When set to True, the tail of the sound
    (e.g. reverb at the end) will be chopped off so that the length of the output is
    equal to the length of the input.

[`padding`](#padding){ #padding }: `float` • unit: meters
:   :octicons-milestone-24: Default: `0.1`. Minimum distance in meters between source or
    mic and the room walls, floor or ceiling.

[`p`](#p){ #p }: `float` • range: [0.0, 1.0]
:   :octicons-milestone-24: Default: `0.5`. The probability of applying this transform.

[`ray_tracing_options`](#ray_tracing_options){ #ray_tracing_options }: `Optional[Dict]`
:   :octicons-milestone-24: Default: `None`. Options for the ray tracer. See `set_ray_tracing` here:  
    [https://github.com/LCAV/pyroomacoustics/blob/master/pyroomacoustics/room.py](https://github.com/LCAV/pyroomacoustics/blob/master/pyroomacoustics/room.py)
