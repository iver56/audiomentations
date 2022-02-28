try:
    import pyroomacoustics as pra
except ImportError:
    print(
        "Error importing `pyroomacoustics`. You should probably run `pip install audiomentations[extra]`"
    )
    raise


import numpy as np
import random

from audiomentations.core.transforms_interface import BaseWaveformTransform


class RoomSimulator(BaseWaveformTransform):
    """
    A ShoeBox Room Simulator. Simulates a cuboid of parametrized size and 
    average surface absorption coefficient. It also includes a source \
    and microphones in parametrized locations.
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
        min_absorption_value: float = 0.15,
        max_absorption_value: float = 0.45,
        min_source_x: float = 0.5,
        max_source_x: float = 0.5,
        min_source_y: float = 0.5,
        max_source_y: float = 0.5,
        min_source_z: float = 0.5,
        max_source_z: float = 0.5,
        min_mic_radius: float = 0.25,
        max_mic_radius: float = 0.5,
        min_mic_azimuth: float = 0.0,
        max_mic_azimuth: float = 3.1419,
        min_mic_elevation: float = 0.0,
        max_mic_elevation: float = 3.1419,
        absorption_mode: float = "absorption",
        use_ray_tracing: float = False,
        max_order: int or None = None,
        p: float = 0.5,
    ):
        """

        :param min_size_x: Minimum width (x coordinate) of the room in meters
        :param max_size_x: Maximum width of the room in meters
        :param min_size_y: Minimum depth (y coordinate) of the room in meters
        :param max_size_y: Maximum depth of the room in meters
        :param min_size_z: Minimum height (z coordinate) of the room in meters
        :param max_size_z: Maximum height of the room in meters
        :param min_absorption_value: Minimum absorption value of the room
            (either average surface absorption coefficient or reverberation time in seconds)
        :param min_absorption_value: Maximum absorption value of the room
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
        :param use_ray_tracing: Whether to use ray_tracing or not (slower but more accurate)
        :param max_order: Maximum order of the Image Source Model. Higher is more accurate but
                          slower. Leave this None if you supply reverberation times instead.
        :param p: The probability of applying this transform
        """
        super().__init__(p)

        assert absorption_mode in [
            "rt60",
            "absorption",
        ], "`absorption_mode` should either be `rt60` or `absorption`"

        self.max_order = max_order
        self.absorption_mode = absorption_mode
        self.min_absorption_value = min_absorption_value
        self.max_absorption_value = max_absorption_value

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

    def randomize_parameters(self, samples: np.array, sample_rate: int):

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

        self.parameters["absorption_coefficient"] = random.uniform(
            self.min_absorption_value, self.max_absorption_value
        )

        self.parameters["max_order"] = self.max_order

        if self.absorption_mode == "rt60":

            # If we are in rt60 mode, estimate the absorption coefficient on a sampled desired
            # rt60 value.
            self.parameters["absorption_coefficient"], max_order = pra.inverse_sabine(
                self.parameters["absorption_coefficient"], room_dim
            )

            # Prioritise manually set `max_order` if it is set, over the one
            # calculated by the inverse sabine formula.
            if not self.max_order:
                self.parameters["max_order"] = max_order

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
        assert samples.dtype == np.float32

        # It makes no sense to use stereo data.
        if len(samples.shape) > 1:
            if samples.shape[0] > samples.shape[1]:
                samples = samples.mean(1)
            else:
                samples = samples.mean(0)

        # Construct room
        room = pra.Room.from_corners(
            np.array(
                [
                    [0, 0],
                    [0, self.parameters["size_x"]],
                    [self.parameters["size_x"], self.parameters["size_y"]],
                ]
            ),
            fs=sample_rate,
            materials=pra.Material(self.parameters["absorption_coefficient"]),
            ray_tracing=self.use_ray_tracing,
            air_absorption=True,
        )

        if self.use_ray_tracing:
            # TODO: Somehow make those parameters
            room.set_ray_tracing(receiver_radius=0.5, n_rays=10000, energy_thres=1e-5)

        room.extrude(
            height=self.parameters["size_z"],
            materials=pra.Material(self.parameters["absorption_coefficient"]),
        )

        # Add the point source
        room.add_source(
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
        room.add_microphone_array(
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
                room.fs,
            )
        )

        # Do the simulation
        room.compute_rir()

        # Store a measured rt60 parameter
        self.parameters["theoretical_rt60"] = room.rt60_theory()
        self.parameters["measured_rt60"] = room.measure_rt60()

        room.simulate()

        return room.mic_array.signals
