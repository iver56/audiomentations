# Audiomentations

A python library for doing audio data augmentation

# Setup

`pip install audiomentations`

# Usage example

```python
from audiomentations import Compose, AddGaussianNoise
import numpy as np

SAMPLE_RATE = 16000

augmenter = Compose([
    AddGaussianNoise(p=0.1)
])

samples = np.zeros((20,), dtype=np.float32)
samples = augmenter(samples=samples, sample_rate=SAMPLE_RATE)
```
