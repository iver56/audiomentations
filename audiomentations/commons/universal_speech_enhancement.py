from audiomentations.core.composition import SomeOf, OneOf

from audiomentations import (
    BandPassFilter,
    HighPassFilter,
    LowPassFilter,
    BandLimitWithTwoPhaseResample,

    ApplyMP3Codec,
    ApplyVorbisCodec,
    ApplyULawCodec,

    Overdrive,
    ClippingDistortion,

    Compressor,
    DestroyLevels,
    NoiseGate,
    SimpleCompressor,
    SimpleExpansor,
    Tremolo,

    BandStopFilter,
    SevenBandParametricEQ,
    TwoPoleAllPassFilter,

    AddBackgroundNoise,
    AddShortNoises,

    Phaser,
    ApplyImpulseResponse,
    ShortDelay,

    AddGaussianNoise
)

def universal_speech_enhancement(environmental_noises_path, background_noises_path, short_noises_path, impulse_responses_path):
    # Implementation of the universal speech enhancement augmentation from https://arxiv.org/pdf/2206.03065.pdf
    augment = SomeOf(
        num_transforms=([1, 2, 3, 4, 5], [0.35, 0.45, 0.15, 0.04, 0.01]),
        weights=[1, 1, 1, 1, 1, 4, 1, 1],
        transforms=[
            OneOf([
                BandPassFilter(p=1),
                HighPassFilter(p=1),
                LowPassFilter(p=1),
                BandLimitWithTwoPhaseResample(p=1),
            ], weights=[5, 5, 20, 30]),
            OneOf([
                ApplyVorbisCodec(p=1),
                ApplyULawCodec(p=1)
            ], weights=[3, 3]),
            OneOf([
                Overdrive(p=1),
                ClippingDistortion(p=1)
            ], weights=[5, 8]),
            OneOf([
                Compressor(p=1, max_makeup=3),
                DestroyLevels(p=1),
                NoiseGate(p=1),
                SimpleCompressor(p=1),
                SimpleExpansor(p=1),
                Tremolo(p=1)
            ], weights=[10, 20, 10, 3, 2, 2]),
            OneOf([
                BandStopFilter(p=1),
                SevenBandParametricEQ(p=1),
                TwoPoleAllPassFilter(p=1)
            ]),
            SomeOf(
                num_transforms=([1, 2, 3], [0.2, 0.5, 0.3]),
                transforms=[
                    AddBackgroundNoise(environmental_noises_path, p=1),
                    AddBackgroundNoise(background_noises_path, p=1),
                    AddShortNoises(short_noises_path, noise_rms='relative_to_whole_input', p=1)
                ]
            ),
            OneOf([
                Phaser(p=1),
                ApplyImpulseResponse(impulse_responses_path, p=1),
                ShortDelay(p=1)
            ], weights=[1, 120, 3]),
            OneOf([
                AddGaussianNoise(p=1)
            ])
        ]
    )

    return augment