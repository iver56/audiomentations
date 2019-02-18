# Audiomentations

![Build status](https://codeship.com/projects/d192b290-158e-0137-8d9a-32050e1fba78/status?branch=master)

A python library for doing audio data augmentation

# Setup

![PyPI version](https://img.shields.io/pypi/v/audiomentations.svg?style=flat)
![Number of downloads from PyPI per month](https://img.shields.io/pypi/dm/audiomentations.svg?style=flat)

`pip install audiomentations`

# Usage example

```python
from audiomentations import Compose, AddGaussianNoise, TimeStretch
import numpy as np

SAMPLE_RATE = 16000

augmenter = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
])

samples = np.zeros((20,), dtype=np.float32)
samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
```

# Development

## Code style

Format the code with `black`

## Run tests

`nosetests`
