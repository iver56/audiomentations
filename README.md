# Audiomentations

![Build status](https://codeship.com/projects/d192b290-158e-0137-8d9a-32050e1fba78/status?branch=master)

A python library for doing audio data augmentation

# Setup

`pip install audiomentations`

# Usage example

```python
from audiomentations import Compose, AddGaussianNoise
import numpy as np

SAMPLE_RATE = 16000

augmenter = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.006, p=0.1)
])

samples = np.zeros((20,), dtype=np.float32)
samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
```

# Development

## Code style

Format the code with `black`

## Run tests

`nosetests`
