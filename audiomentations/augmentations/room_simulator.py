import sys
from typing import Optional
import numpy as np
import random

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

    > augment = RoomSimulator(calculate_by_absorption_or_rt60="rt60", min_absorption_value_or_rt60=0.06, max_absorption_value_or_rt60=0.06, min_size_x = ...)
    Augment with randomly selected room impulse responses that have an RT60 of 0.06.

    > augment = RoomSimulator(min_mic_radius=1.0, max_min_radius=1.0)
    Augment with a RIR captured by all positions of the microphone on a sphere, centred around the source at 1m

    > augment = RoomSimulator(min_mic_radius=1.0, max_min_radius=1.0, min_mic_elevation=0.0, max_mic_elevation=0.0)
    Augment with a RIR captured by all positions of the microphone on a circle, centred around the source at 1m
    
    Additionally, RoomSimulator.parameters["measured_rt60"] can then be used to map augmented audio to measured rt60 (for e.g. blind RT60 estimation using deep learning)
    """

    supports_multichannel = True

    def __init__(
        self,
        min_size_x: float = 1.0,
        max_size_x: float = 1.0,
        min_size_y: float = 1.0,
        max_size_y: float = 1.0,
        min_size_z: float = 1.0,
        max_size_z: float = 1.0,
        min_absorption_value_or_rt60: float = 0.15,
        max_absorption_value_or_rt60: float = 0.45,
        min_source_x: float = 0.5,
        max_source_x: float = 0.5,
        min_source_y: float = 0.5,
        max_source_y: float = 0.5,
        min_source_z: float = 0.5,
        max_source_z: float = 0.5,
        min_mic_radius: float = 0.25,
        max_mic_radius: float = 0.5,
        min_mic_azimuth: float = -3.1419,
        max_mic_azimuth: float = 3.1419,
        min_mic_elevation: float = -3.1419,
        max_mic_elevation: float = 3.1419,
        calculate_by_absorption_or_rt60: float = "absorption",
        use_ray_tracing: float = True,
        max_order: int or None = None,
        leave_length_unchanged: Optional[bool] = None,
        p: float = 0.5,
    ):
        """

        :param min_size_x: Minimum width (x coordinate) of the room in meters
        :param max_size_x: Maximum width of the room in meters
        :param min_size_y: Minimum depth (y coordinate) of the room in meters
        :param max_size_y: Maximum depth of the room in meters
        :param min_size_z: Minimum height (z coordinate) of the room in meters
        :param max_size_z: Maximum height of the room in meters
        :param min_absorption_value_or_rt60: Minimum absorption or rt60 of the room
            (either average surface absorption coefficient or reverberation time in seconds
            depending on the value of `calculate_by_absorption_or_rt60`)
        :param max_absorption_value_or_rt60: Minimum absorption or rt60 of the room
        :param min_source_x: Minimum x location of the source (meters)
        :param max_source_x: Minimum x location of the source (meters)
        :param min_source_y:
        :param max_source_y:
        :param min_source_z:
        :param max_source_z:
        :param min_mic_radius: Minimum distance of the microphone from the source in meters
        :param max_mic_radius:
        :param min_mic_azimuth: Minimum azimuth (angle around z axis) of the microphone
                                relative to the source, in radians.
        :param max_mic_azimuth:
        :param min_mic_elevation:
                                Minimum elevation of the microphon relative to the source, in
                                radians.
        :param max_mic_elevation:
        :param use_ray_tracing: Whether to use ray_tracing or not (slower but much more accurate).
                          Disable this if you need speed but do not really care for incorrect results.
        :param max_order: Maximum order of the Image Source Model. Higher is more accurate but
                          slower. Leave this None if you supply reverberation times instead.
        :param leave_length_unchanged: When set to True, the tail of the sound (e.g. reverb at
            the end) will be chopped off so that the length of the output is equal to the
            length of the input.
        :param p: The probability of applying this transform
        """
        super().__init__(p)

        assert calculate_by_absorption_or_rt60 in [
            "rt60",
            "absorption",
        ], "`calculate_by_absorption_or_rt60` should either be `rt60` or `absorption`"

        self.max_order = max_order
        self.absorption_mode = calculate_by_absorption_or_rt60
        self.min_absorption_value_or_rt60 = min_absorption_value_or_rt60
        self.max_absorption_value_or_rt60 = max_absorption_value_or_rt60

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

        assert min_mic_radius <= max_mic_radius
        assert min_mic_azimuth <= max_mic_azimuth
        assert min_mic_elevation <= max_mic_elevation

        self.min_mic_radius = min_mic_radius
        self.max_mic_radius = max_mic_radius

        self.min_mic_azimuth = min_mic_azimuth
        self.max_mic_azimuth = max_mic_azimuth

        self.min_mic_elevation = min_mic_elevation
        self.max_mic_elevation = max_mic_elevation

        self.leave_length_unchanged = leave_length_unchanged

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

        # absorption_value_or_rt60 is the absorption coefficient when the mode is `absorption`
        # otherwise it is `rt60`
        absorption_value_or_rt60 = random.uniform(
            self.min_absorption_value_or_rt60, self.max_absorption_value_or_rt60
        )

        self.parameters["max_order"] = self.max_order

        if self.absorption_mode == "rt60":
            self.parameters["target_rt60"] = absorption_value_or_rt60

            # If we are in rt60 mode, estimate the absorption coefficient on a sampled desired
            # rt60 value.
            self.parameters["absorption_coefficient"], max_order = pra.inverse_sabine(
                self.parameters["target_rt60"], room_dim
            )

            # Prioritise manually set `max_order` if it is set, over the one
            # calculated by the inverse sabine formula.
            if not self.max_order:
                self.parameters["max_order"] = max_order
        else:
            self.parameters["absorption_coefficient"] = absorption_value_or_rt60

        self.parameters["source_x"] = random.uniform(
            self.min_source_x, self.max_source_x
        )
        self.parameters["source_y"] = random.uniform(
            self.min_source_y, self.max_source_y
        )
        self.parameters["source_z"] = random.uniform(
            self.min_source_z, self.max_source_z
        )

        self.parameters["mic_radius"] = random.uniform(
            self.min_mic_radius, self.max_mic_radius
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
        self.parameters["mic_x"] = max(0, min(self.parameters["size_x"], mic_x))
        self.parameters["mic_y"] = max(0, min(self.parameters["size_y"], mic_y))
        self.parameters["mic_z"] = max(0, min(self.parameters["size_z"], mic_z))

    def apply(self, samples, sample_rate):
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

        assert samples.dtype == np.float32

        # It makes no sense to use stereo data.
        if len(samples.shape) > 1:
            if samples.shape[0] > samples.shape[1]:
                samples = samples.mean(1)
            else:
                samples = samples.mean(0)

        # Construct room
        self.room = pra.Room.from_corners(
            np.array(
                [
                    [0, 0],
                    [0, self.parameters["size_x"]],
                    [self.parameters["size_x"], self.parameters["size_y"]],
                ]
            ).T,
            fs=sample_rate,
            materials=pra.Material(self.parameters["absorption_coefficient"]),
            ray_tracing=self.use_ray_tracing,
            air_absorption=True,
        )

        if self.use_ray_tracing:
            # TODO: Somehow make those parameters
            self.room.set_ray_tracing(
                receiver_radius=0.5, n_rays=10000, energy_thres=1e-5
            )

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
        self.room.simulate()
        result = self.room.mic_array.signals.astype(np.float32).flatten()

        num_samples = max(samples.shape)

        if self.leave_length_unchanged:
            return result[:num_samples]
        return result
