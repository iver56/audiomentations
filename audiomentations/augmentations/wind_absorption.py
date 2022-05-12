from audiomentations.core.transforms_interface import BaseWaveformTransform
import numpy as np
import librosa


class WindAbsorption(BaseWaveformTransform):
    """
    Applies a wind absorption transform parametrized by temperature, humidity, and distance
    of source.

    This is not a scientifically accurate transform but basically applies a uniform
    filterbank with attenuations given by:

    att = exp(- distance * absorption_coefficient)

    where distance is the microphone-source assumed distance in meters and `absorption_coefficient`
    is adapted from a look up table by pyroomacoustics [1].

    Note: This only "simulates" room absorption, and does not apply the distance law
    (use a GainTransform for that)

    [1] https://github.com/LCAV/pyroomacoustics
    """

    supports_multichannel = True

    # Table of air absorption coefficients adapted from `pyroomacoustics`.
    # The last two columns have been extrapolated with `curve_fit`
    air_absorption_table = {
        "10C_30-50%": [
            x * 1e-3 for x in [0.1, 0.2, 0.5, 1.1, 2.7, 9.4, 29.0, 91.5, 289.0]
        ],
        "10C_50-70%": [
            x * 1e-3 for x in [0.1, 0.2, 0.5, 0.8, 1.8, 5.9, 21.1, 76.6, 280.2]
        ],
        "10C_70-90%": [
            x * 1e-3 for x in [0.1, 0.2, 0.5, 0.7, 1.4, 4.4, 15.8, 58.0, 214.9]
        ],
        "20C_30-50%": [
            x * 1e-3 for x in [0.1, 0.3, 0.6, 1.0, 1.9, 5.8, 20.3, 72.3, 259.9]
        ],
        "20C_50-70%": [
            x * 1e-3 for x in [0.1, 0.3, 0.6, 1.0, 1.7, 4.1, 13.5, 44.4, 148.7]
        ],
        "20C_70-90%": [
            x * 1e-3 for x in [0.1, 0.3, 0.6, 1.1, 1.7, 3.5, 10.6, 31.2, 93.8]
        ],
        "center_freqs": [125, 250, 500, 1000, 2000, 4000, 8000, 16000, 32000],
    }

    def __init__(
        self,
        min_temperature=10.0,
        max_temperature=10.0,
        min_humidity=30,
        max_humidity=90,
        min_distance=10,
        max_distance=100,
        n_fft=256,
        p=0.5,
    ):
        """
        :param min_temperature: Minimum temperature in celcius
        :param max_temperature: Maximum temperature in celsius
        :param min_humidity: Minimum humidity in percent %
        :param max_humidity: Maximum humidity in percent %
        :param min_distance: Minimum microphone-source distance in meters.
        :param max_distance: Maximum microphone-source distance in meters.
        :param n_fft: The number of bins in the estimated filterbank
        :param p: The probability of applying this transform
        """
        assert float(min_temperature) in [
            10.0,
            20.0,
        ], "Sorry, the only supported temperatures are either 10 or 20 degrees celcius"
        assert float(max_temperature) in [
            10.0,
            20.0,
        ], "Sorry, the only supported temperatures are either 10 or 20 degrees celcius"
        assert min_temperature <= max_temperature
        assert 30 <= min_humidity <= max_humidity <= 90

        super().__init__(p)

        self.min_temperature = min_temperature
        self.max_temperature = max_temperature

        self.min_humidity = min_humidity
        self.max_humidity = max_humidity

        self.min_distance = min_distance
        self.max_distance = max_distance

        self.n_fft = n_fft

    def randomize_parameters(self, samples, sample_rate):
        super().randomize_parameters(samples, sample_rate)
        self.parameters["temperature"] = 10 * np.random.randint(
            int(self.min_temperature) // 10, int(self.max_temperature) // 10 + 1
        )
        self.parameters["humidity"] = np.random.randint(
            self.min_humidity, self.max_humidity + 1
        )
        self.parameters["distance"] = np.random.uniform(
            self.min_distance, self.max_distance
        )

    def apply(self, samples, sample_rate):
        assert samples.dtype == np.float32

        humidity = self.parameters["humidity"]
        distance = self.parameters["distance"]

        # Choose correct absorption coefficients
        key = str(int(self.parameters["temperature"])) + "C"
        bounds = [30, 50, 70, 90]
        for n in range(1, len(bounds)):
            if bounds[n - 1] <= humidity or humidity <= bounds[n]:
                key += f"_{bounds[n-1]}-{bounds[n]}%"
                break

        # Convert to attenuations
        attenuation_values = np.exp(
            -distance * np.array(self.air_absorption_table[key])
        )

        # Frequencies to calculate the attenuations caused by wind absorption
        frequencies = librosa.fft_frequencies(sr=48000, n_fft=256)

        # Interpolate to the desired frequencies (we have to do this in dB)
        db_target_attenuations = np.interp(
            frequencies,
            self.air_absorption_table["center_freqs"],
            20 * np.log10(attenuation_values),
        )

        linear_target_attenuations = 10 ** (db_target_attenuations / 20)

        result = np.zeros_like(samples)

        # Apply using STFT
        if len(samples.shape) == 1:
            stft = librosa.stft(samples, n_fft=self.n_fft)

            # Compute mask
            mask = np.tile(linear_target_attenuations, (stft.shape[1], 1)).T

            # Compute target degraded audio
            degraded_audio = librosa.istft(stft * mask)

            min_length = min(len(degraded_audio), len(result))
            result[:min_length] = degraded_audio[:min_length]

        else:
            for chn_idx, channel in enumerate(samples):
                stft = librosa.stft(channel, n_fft=self.n_fft)

                # Compute mask
                mask = np.tile(linear_target_attenuations, (stft.shape[1], 1)).T

                # Compute target degraded audio
                degraded_channel = librosa.istft(stft * mask)

                min_length = min(len(degraded_channel), result.shape[1])
                result[chn_idx, :min_length] = degraded_channel

        return result
