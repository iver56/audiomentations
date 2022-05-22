import random
from typing import Optional, Dict

import numpy as np
import sys
from scipy.signal import convolve

from audiomentations.core.transforms_interface import BaseWaveformTransform


class RoomSimulator(BaseWaveformTransform):
    """
    A ShoeBox Room Simulator. Simulates a cuboid of parametrized size and 
    average surface absorption coefficient. It also includes a source \
    and microphones in parametrized locations.

    Use it when you want a ton of synthetic room impulse responses of specific configurations
    characteristics or simply to quickly add reverb for augmentation purposes

    Some examples:

    > augment = RoomSimulator(use_ray_tracing=False)
    Quickly add reverberation without caring for accuracy.

    > augment = RoomSimulator(calculation_mode="rt60", min_target_rt60=0.06, max_target_rt60=0.06, min_size_x = ...)
    Augment with randomly selected room impulse responses that have an RT60 of 0.06.

    > augment = RoomSimulator(min_mic_radius=1.0, max_min_radius=1.0)
    Augment with a RIR captured by all positions of the microphone on a sphere, centred around the source at 1m

    > augment = RoomSimulator(min_mic_radius=1.0, max_min_radius=1.0, min_mic_elevation=0.0, max_mic_elevation=0.0)
    Augment with a RIR captured by all positions of the microphone on a circle, centred around the source at 1m    
    """

    supports_multichannel = True

    def __init__(
        self,
        min_size_x: float = 3.6,
        max_size_x: float = 5.6,
        min_size_y: float = 3.6,
        max_size_y: float = 3.9,
        min_size_z: float = 2.4,
        max_size_z: float = 3.0,
        min_absorption_value: float = 0.075,
        max_absorption_value: float = 0.4,
        min_target_rt60: float = 0.15,
        max_target_rt60: float = 0.8,
        min_source_x: float = 0.1,
        max_source_x: float = 3.5,
        min_source_y: float = 0.1,
        max_source_y: float = 2.7,
        min_source_z: float = 1.0,
        max_source_z: float = 2.1,
        min_mic_distance: float = 0.15,
        max_mic_distance: float = 0.35,
        min_mic_azimuth: float = -np.pi,
        max_mic_azimuth: float = np.pi,
        min_mic_elevation: float = -np.pi,
        max_mic_elevation: float = np.pi,
        calculation_mode: float = "absorption",
        use_ray_tracing: float = True,
        max_order: Optional[int] = None,
        leave_length_unchanged: Optional[bool] = None,
        padding: float = 0.1,
        p: float = 0.5,
        ray_tracing_options: Dict or None = None,
    ):
        """

        :param min_size_x: Minimum width (x coordinate) of the room in meters
        :param max_size_x: Maximum width of the room in meters
        :param min_size_y: Minimum depth (y coordinate) of the room in meters
        :param max_size_y: Maximum depth of the room in meters
        :param min_size_z: Minimum height (z coordinate) of the room in meters
        :param max_size_z: Maximum height of the room in meters
        :param min_absorption_value: When `calculation_mode` is 'absorption' it will set
            a given coefficient value for the surfaces of the room (walls, ceilings, and floor).
            This coefficient takes values between 0 (fully reflective surface) and 1 (fully absorbing surface).

            Example values (May differ!):
                Studio w acoustic panels > 0.40
                Office / Library         ~ 0.15
                Factory                  ~ 0.05
        :param max_absorption_value:

        :param min_target_rt60: When `calculation_mode` is `rt60`, it tries to set the absorption value
            of the surfaces of the room to achieve a target rt60 (in seconds). Note that this parameter
            changes only the materials (absorption coefficients) of the surfaces, NOT the dimension of the rooms.

            Example values (May differ!):
                Recording studio:  0.3s
                Office          :  0.5s
                Concert Hall    :  1.5s

        :param min_source_x: Minimum x location of the source (meters)
        :param max_source_x: Minimum x location of the source (meters)
        :param min_source_y:
        :param max_source_y:
        :param min_source_z:
        :param max_source_z:
        :param min_mic_distance: Minimum distance of the microphone from the source in meters
        :param max_mic_distance:
        :param min_mic_azimuth: Minimum azimuth (angle around z axis) of the microphone
            relative to the source, in radians.
        :param max_mic_azimuth:
        :param min_mic_elevation:
            Minimum elevation of the microphon relative to the source, in
            radians.
        :param max_mic_elevation:
        :param calculation_mode: When set to `absorption`, it will create the room with surfaces based on
            `min_absorption_value` and `max_absorption_value`. If set to `rt60` it will try to assign surface
            materials that lead to a room impulse response with target rt60 given by `min_target_rt60` and `max_target_rt60`
        :param use_ray_tracing: Whether to use ray_tracing or not (slower but much more accurate).
            Disable this if you need speed but do not really care for incorrect results.
        :param max_order: Maximum order of reflections for the Image Source Model. E.g. a value of
            1 will only add first order reflections while a value of 30 will add a
            diffuse reverberation tail.
        :param leave_length_unchanged: When set to True, the tail of the sound (e.g. reverb at
            the end) will be chopped off so that the length of the output is equal to the
            length of the input.
        :param padding: Minimum distance in meters between source or mic and the room walls, floor or ceiling.
        :param p: The probability of applying this transform
        :param ray_tracing_options: Options for the ray tracer. See `set_ray_tracing` here:
            https://github.com/LCAV/pyroomacoustics/blob/master/pyroomacoustics/room.py
        """
        super().__init__(p)

        assert calculation_mode in [
            "rt60",
            "absorption",
        ], "`calculation_mode` should either be `rt60` or `absorption`"

        self.max_order = max_order
        self.calculation_mode = calculation_mode
        self.min_absorption_value = min_absorption_value
        self.max_absorption_value = max_absorption_value

        self.min_target_rt60 = min_target_rt60
        self.max_target_rt60 = max_target_rt60

        self.use_ray_tracing = use_ray_tracing
        self.max_order = max_order

        self.min_size_x = min_size_x
        self.min_size_y = min_size_y
        self.min_size_z = min_size_z

        self.max_size_x = max_size_x
        self.max_size_y = max_size_y
        self.max_size_z = max_size_z

        assert min_source_x <= max_source_x
        assert min_source_y <= max_source_y
        assert min_source_z <= max_source_z

        self.min_source_x = min_source_x
        self.max_source_x = max_source_x

        self.min_source_y = min_source_y
        self.max_source_y = max_source_y

        self.min_source_z = min_source_z
        self.max_source_z = max_source_z

        assert min_mic_distance <= max_mic_distance
        assert min_mic_azimuth <= max_mic_azimuth
        assert min_mic_elevation <= max_mic_elevation

        self.min_mic_distance = min_mic_distance
        self.max_mic_distance = max_mic_distance

        self.min_mic_azimuth = min_mic_azimuth
        self.max_mic_azimuth = max_mic_azimuth

        self.min_mic_elevation = min_mic_elevation
        self.max_mic_elevation = max_mic_elevation

        self.leave_length_unchanged = leave_length_unchanged

        self.padding = padding

        if ray_tracing_options is None:
            self.ray_tracing_options = {
                "receiver_radius": 0.5,
                "n_rays": 10000,
                "energy_thres": 1e-5,
            }
        else:
            self.ray_tracing_options = ray_tracing_options

    def randomize_parameters(self, samples: np.array, sample_rate: int):

        try:
            import pyroomacoustics as pra
        except ImportError:

            print(
                "Failed to import pyroomacoustics. Maybe it is not installed? "
                "To install the optional pyroomacoustics dependency of audiomentations,"
                " do `pip install audiomentations[extras]` or simply "
                " `pip install pyroomacoustics`",
                file=sys.stderr,
            )
            raise

        super().randomize_parameters(samples, sample_rate)
        self.parameters["size_x"] = random.uniform(self.min_size_x, self.max_size_x)
        self.parameters["size_y"] = random.uniform(self.min_size_y, self.max_size_y)
        self.parameters["size_z"] = random.uniform(self.min_size_z, self.max_size_z)

        room_dim = np.array(
            [
                self.parameters["size_x"],
                self.parameters["size_y"],
                self.parameters["size_z"],
            ]
        )

        self.parameters["max_order"] = self.max_order

        if self.calculation_mode == "rt60":
            target_rt60 = random.uniform(self.min_target_rt60, self.max_target_rt60)
            self.parameters["target_rt60"] = target_rt60

            # If we are in rt60 mode, estimate the absorption coefficient on a desired target
            # rt60 value.
            self.parameters["absorption_coefficient"], max_order = pra.inverse_sabine(
                self.parameters["target_rt60"], room_dim
            )

            # Prioritise manually set `max_order` if it is set, over the one
            # calculated by the inverse sabine formula.
            if not self.max_order:
                self.parameters["max_order"] = max_order
        else:
            self.parameters["absorption_coefficient"] = random.uniform(
                self.min_absorption_value, self.max_absorption_value
            )

        self.parameters["source_x"] = random.uniform(
            max(self.min_source_x, self.padding),
            min(self.max_source_x, self.parameters["size_x"] - self.padding),
        )
        self.parameters["source_y"] = random.uniform(
            max(self.min_source_y, self.padding),
            min(self.max_source_y, self.parameters["size_y"] - self.padding),
        )
        self.parameters["source_z"] = random.uniform(
            max(self.min_source_z, self.padding),
            min(self.max_source_z, self.parameters["size_z"] - self.padding),
        )

        self.parameters["mic_radius"] = random.uniform(
            self.min_mic_distance, self.max_mic_distance
        )
        self.parameters["mic_azimuth"] = random.uniform(
            self.min_mic_azimuth, self.max_mic_azimuth
        )
        self.parameters["mic_elevation"] = random.uniform(
            self.min_mic_elevation, self.max_mic_elevation
        )

        # Convert to cartesian coordinates according to ADM
        mic_x = self.parameters["source_x"] - self.parameters["mic_radius"] * np.cos(
            self.parameters["mic_elevation"]
        ) * np.sin(self.parameters["mic_azimuth"])
        mic_y = self.parameters["source_y"] + self.parameters["mic_radius"] * np.cos(
            self.parameters["mic_elevation"]
        ) * np.cos(self.parameters["mic_azimuth"])
        mic_z = self.parameters["source_z"] + self.parameters["mic_radius"] * np.sin(
            self.parameters["mic_elevation"]
        )

        # Clamp between 0 and room dimensions
        self.parameters["mic_x"] = max(
            self.padding, min(self.parameters["size_x"] - self.padding, mic_x)
        )
        self.parameters["mic_y"] = max(
            self.padding, min(self.parameters["size_y"] - self.padding, mic_y)
        )
        self.parameters["mic_z"] = max(
            self.padding, min(self.parameters["size_z"] - self.padding, mic_z)
        )

        # Construct room
        self.room = pra.Room.from_corners(
            np.array(
                [
                    [0, 0],
                    [0, self.parameters["size_x"]],
                    [self.parameters["size_x"], self.parameters["size_y"]],
                    [self.parameters["size_y"], 0],
                ]
            ).T,
            fs=sample_rate,
            materials=pra.Material(self.parameters["absorption_coefficient"]),
            ray_tracing=self.use_ray_tracing,
            air_absorption=True,
        )

        if self.use_ray_tracing:
            # TODO: Somehow make those parameters
            self.room.set_ray_tracing(**self.ray_tracing_options)

        self.room.extrude(
            height=self.parameters["size_z"],
            materials=pra.Material(self.parameters["absorption_coefficient"]),
        )

        # Add the point source
        self.room.add_source(
            np.array(
                [
                    self.parameters["source_x"],
                    self.parameters["source_y"],
                    self.parameters["source_z"],
                ]
            ),
            signal=samples,
        )

        # Add the microphone
        self.room.add_microphone_array(
            pra.MicrophoneArray(
                np.array(
                    [
                        [
                            self.parameters["mic_x"],
                            self.parameters["mic_y"],
                            self.parameters["mic_z"],
                        ]
                    ]
                ).T,
                self.room.fs,
            )
        )

        # Do the simulation
        self.room.compute_rir()

    def apply(self, samples, sample_rate):
        assert samples.dtype == np.float32

        rir = self.room.rir[0][0]

        # This is the same as ApplyImpulseResponse transform
        if samples.ndim > 1:
            signal_ir = []
            for i in range(samples.shape[0]):
                channel_conv = convolve(samples[i], rir)
                signal_ir.append(channel_conv)
            signal_ir = np.array(signal_ir, dtype=samples.dtype)
        else:
            signal_ir = convolve(samples, rir).astype(samples.dtype)

        if self.leave_length_unchanged:
            signal_ir = signal_ir[..., : samples.shape[-1]]
        return signal_ir
