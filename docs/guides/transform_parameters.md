# Transform parameters

## How to obtain the chosen parameters after calling a transform

You can access the `parameters` property of a transform. Code example:

```python
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import numpy as np

augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
    Shift(p=0.5),
])

# Generate 2 seconds of dummy audio for the sake of example
samples = np.random.uniform(low=-0.2, high=0.2, size=(32000,)).astype(np.float32)

# Augment/transform/perturb the audio data
augmented_samples = augment(samples=samples, sample_rate=16000)

for transform in augment.transforms:
    print(f"{transform.__class__.__name__}: {transform.parameters}")
```

When running the example code above, it may print something like this:
```
AddGaussianNoise: {'should_apply': True, 'amplitude': 0.0027702725003923272}
TimeStretch: {'should_apply': True, 'rate': 1.158377360016495}
PitchShift: {'should_apply': False}
Shift: {'should_apply': False}
```

## How to use apply a transform with the same parameters to multiple inputs

This technique can be useful if you want to transform e.g. a target sound in the same way as an input sound. Code example:

```python
from audiomentations import Gain
import numpy as np

augment = Gain(p=1.0)

samples = np.random.uniform(low=-0.2, high=0.2, size=(32000,)).astype(np.float32)
samples2 = np.random.uniform(low=-0.2, high=0.2, size=(32000,)).astype(np.float32)

augmented_samples = augment(samples=samples, sample_rate=16000)
augment.freeze_parameters()
print(augment.parameters)
augmented_samples2 = augment(samples=samples2, sample_rate=16000)
print(augment.parameters)
augment.unfreeze_parameters()
```

When running the example code above, it may print something like this:

```
{'should_apply': True, 'amplitude_ratio': 0.9688148624484364}
{'should_apply': True, 'amplitude_ratio': 0.9688148624484364}
```

In other words, this means that both sounds (`samples` and `samples2`) were gained by the same amount

### Freezing a composition

`Compose.freeze_parameters()` preserves the composition's decision to apply or skip
the transforms (based on `p`) and their order when `shuffle=True`. By default, it also
freezes the child transforms, including nested compositions. Calling
`unfreeze_parameters()` allows these parameters to be randomized again on the next call.

If a `Compose` is frozen before its first call, that call chooses its parameters once
and subsequent calls reuse them, just like individual waveform transforms.

Use `apply_to_children=False` with `freeze_parameters()` or `unfreeze_parameters()` to
change only the composition's frozen state. For example, freezing only a `Compose`
preserves its execution decision and order while its unfrozen children continue to
choose their own parameters.

An explicit call to `randomize_parameters(samples, sample_rate)` chooses new parameters
even while frozen, including for the children by default. Pass `apply_to_children=False`
to randomize only the composition's decision and order.
