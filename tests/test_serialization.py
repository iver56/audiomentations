import json
import os
import pickle

import numpy as np
import pytest

from audiomentations import (
    AddBackgroundNoise,
    AddColorNoise,
    AddGaussianNoise,
    AddGaussianSNR,
    AddShortNoises,
    AdjustDuration,
    AirAbsorption,
    Aliasing,
    ApplyImpulseResponse,
    BandPassFilter,
    BandStopFilter,
    BitCrush,
    Clip,
    ClippingDistortion,
    Gain,
    GainTransition,
    HighPassFilter,
    HighShelfFilter,
    Lambda,
    Limiter,
    LoudnessNormalization,
    LowPassFilter,
    LowShelfFilter,
    Mp3Compression,
    Normalize,
    Padding,
    PeakingFilter,
    PitchShift,
    PolarityInversion,
    RepeatPart,
    Resample,
    Reverse,
    RoomSimulator,
    SevenBandParametricEQ,
    Shift,
    TanhDistortion,
    TimeMask,
    TimeStretch,
    Trim,
)
from demo.demo import DEMO_DIR


@pytest.mark.parametrize(
    "transform",
    [
        AddBackgroundNoise(
            sounds_path=os.path.join(DEMO_DIR, "background_noises"), p=1.0
        ),
        AddColorNoise(p=1.0),
        AddGaussianNoise(p=1.0),
        AddGaussianSNR(p=1.0),
        AddShortNoises(sounds_path=os.path.join(DEMO_DIR, "background_noises"), p=1.0),
        AdjustDuration(duration_samples=500, p=1.0),
        AirAbsorption(p=1.0),
        Aliasing(p=1.0),
        ApplyImpulseResponse(ir_path=os.path.join(DEMO_DIR, "ir"), p=1.0),
        BandPassFilter(p=1.0),
        BandStopFilter(p=1.0),
        BitCrush(p=1.0),
        Clip(p=1.0),
        ClippingDistortion(p=1.0),
        Gain(p=1.0),
        GainTransition(p=1.0),
        HighPassFilter(p=1.0),
        HighShelfFilter(p=1.0),
        Lambda(transform=Gain(p=1.0), p=1.0),
        Limiter(p=1.0),
        LoudnessNormalization(p=1.0),
        LowPassFilter(p=1.0),
        LowShelfFilter(p=1.0),
        Mp3Compression(p=1.0),
        Normalize(p=1.0),
        Padding(p=1.0),
        PeakingFilter(p=1.0),
        PitchShift(p=1.0),
        PolarityInversion(p=1.0),
        RepeatPart(p=1.0),
        Resample(p=1.0),
        Reverse(p=1.0),
        # RoomSimulator(p=1.0),  # TODO: Fix TypeError: cannot pickle 'pyroomacoustics.libroom.Wall' object
        SevenBandParametricEQ(p=1.0),
        Shift(p=1.0),
        TanhDistortion(p=1.0),
        TimeMask(p=1.0),
        TimeStretch(p=1.0),
        Trim(p=1.0),
    ],
    ids=lambda transform: transform.__class__.__name__,
)
@pytest.mark.parametrize(
    "samples", [np.random.normal(0, 1, size=17640).astype(np.float32)]
)
def test_param_json_serialization_and_transform_picklability(transform, samples):
    transform.randomize_parameters(samples, sample_rate=44100)
    json.dumps(transform.serialize_parameters())

    # Picklability is required for multiprocessing compatibility
    pickled = pickle.dumps(transform)
    pickle.loads(pickled)
