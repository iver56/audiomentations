import random
import numpy as np


class BaseCompose:
    def __init__(self, transforms, p: float = 1.0, shuffle: bool = False):
        self.transforms = transforms
        self.p = p
        self.shuffle = shuffle
        self.are_parameters_frozen = False

        name_list = []
        for transform in self.transforms:
            name_list.append(type(transform).__name__)
        self.__name__ = "_".join(name_list)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def randomize_parameters(self, *args, **kwargs):
        """
        Randomize and define parameters of every transform in composition.
        """
        apply_to_children = kwargs.get("apply_to_children", True)
        if apply_to_children:
            if "apply_to_children" in kwargs:
                del kwargs["apply_to_children"]
            for transform in self.transforms:
                transform.randomize_parameters(*args, **kwargs)

    def freeze_parameters(self, apply_to_children=True):
        """
        Mark all parameters as frozen, i.e. do not randomize them for each call. This can be
        useful if you want to apply an effect chain with the exact same parameters to multiple
        sounds.
        """
        self.are_parameters_frozen = True
        if apply_to_children:
            for transform in self.transforms:
                transform.freeze_parameters()

    def unfreeze_parameters(self, apply_to_children=True):
        """
        Unmark all parameters as frozen, i.e. let them be randomized for each call.
        """
        self.are_parameters_frozen = False
        if apply_to_children:
            for transform in self.transforms:
                transform.unfreeze_parameters()


class Compose(BaseCompose):
    """
    Compose applies the given sequence of transforms when called,
    optionally shuffling the sequence for every call.

    Example usage:

    ```
    augment = Compose([
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
        PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
        Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
    ])

    # Generate 2 seconds of dummy audio for the sake of example
    samples = np.random.uniform(low=-0.2, high=0.2, size=(32000,)).astype(np.float32)

    # Augment/transform/perturb the audio data
    augmented_samples = augment(samples=samples, sample_rate=16000)
    ```
    """

    def __init__(self, transforms, p=1.0, shuffle=False):
        super().__init__(transforms, p, shuffle)

    def __call__(self, samples, sample_rate):
        transforms = self.transforms.copy()
        should_apply = random.random() < self.p
        # TODO: Adhere to self.are_parameters_frozen
        # https://github.com/iver56/audiomentations/issues/135
        if should_apply:
            if self.shuffle:
                random.shuffle(transforms)
            for transform in transforms:
                samples = transform(samples, sample_rate)

        return samples


class SpecCompose(BaseCompose):
    def __init__(self, transforms, p=1.0, shuffle=False):
        super().__init__(transforms, p, shuffle)

    def __call__(self, magnitude_spectrogram):
        transforms = self.transforms.copy()
        should_apply = random.random() < self.p
        # TODO: Adhere to self.are_parameters_frozen
        # https://github.com/iver56/audiomentations/issues/135
        if should_apply:
            if self.shuffle:
                random.shuffle(transforms)
            for transform in transforms:
                magnitude_spectrogram = transform(magnitude_spectrogram)

        return magnitude_spectrogram



class SomeOf(BaseCompose):
    """
    SomeOf randomly picks one or several of the given transforms when called, and applies these
    transforms.

    INCLUDE EXAMPLES !!!!!!!!

    """

    def __init__(self, transforms, p: float = 1.0):
        super().__init__(transforms, p)
        self.transform_indexes = []
        self.nbr_transforms_to_apply = None
        self.should_apply = True

    def randomize_parameters(self, *args, **kwargs):
        super().randomize_parameters(*args, **kwargs)
        self.should_apply = random.random() < self.p
        if self.should_apply:
            self.nbr_transforms_to_apply = random.randint(1, len(self.transforms))
            all_indexes_transforms = list(np.arange(len(self.transforms)))
            print(all_indexes_transforms)
            self.transform_indexes = sorted(random.sample(all_indexes_transforms, self.nbr_transforms_to_apply))
        return self.transform_indexes

    def __call__(self, *args, **kwargs):
        if not self.are_parameters_frozen:
            kwargs["apply_to_children"] = False
            self.randomize_parameters(*args, **kwargs)

        if self.should_apply:
            if "apply_to_children" in kwargs:
                del kwargs["apply_to_children"]
            print(self.transform_indexes[0])
            transformed_data = self.transforms[self.transform_indexes[0]](*args, **kwargs)

            if hasattr(self.transforms[0], "name"): # Quick fix, means that they are transforms on spectrograms
                for transform_index in self.transform_indexes[1:]:
                    transformed_data = self.transforms[self.transform_indexes[transform_index]](transformed_data)
            else:
                # get access to the sample rate
                if "sample_rate" in kwargs:
                    sample_rate = kwargs["sample_rate"]
                else:
                    sample_rate = args[1]

                for transform_index in self.transform_indexes[1:]:
                    print(transform_index)
                    transformed_data = self.transforms[transform_index](transformed_data, sample_rate)
            return transformed_data

        if "samples" in kwargs:
            return kwargs["samples"]
        elif "magnitude_spectrogram" in kwargs:
            return kwargs["magnitude_spectrogram"]
        else:
            return args[0]


class OneOf(BaseCompose):
    """
    OneOf randomly picks one of the given transforms when called, and applies that
    transform.
    Example usage:
    ```
    augment = OneOf([
        TimeStretch(min_rate=0.8, max_rate=1.25, p=1.0),
        PitchShift(min_semitones=-4, max_semitones=4, p=1.0),
    ])
    # Generate 2 seconds of dummy audio for the sake of example
    samples = np.random.uniform(low=-0.2, high=0.2, size=(32000,)).astype(np.float32)
    # Augment/transform/perturb the audio data
    augmented_samples = augment(samples=samples, sample_rate=16000)
    # Result: The audio was either time-stretched or pitch-shifted, but not both
    ```
    """

    def __init__(self, transforms, p: float = 1.0):
        super().__init__(transforms, p)
        self.transform_index = 0
        self.should_apply = True

    def randomize_parameters(self, *args, **kwargs):
        super().randomize_parameters(*args, **kwargs)
        self.should_apply = random.random() < self.p
        if self.should_apply:
            self.transform_index = random.randint(0, len(self.transforms) - 1)

    def __call__(self, *args, **kwargs):
        if not self.are_parameters_frozen:
            kwargs["apply_to_children"] = False
            self.randomize_parameters(*args, **kwargs)

        if self.should_apply:
            if "apply_to_children" in kwargs:
                del kwargs["apply_to_children"]
            return self.transforms[self.transform_index](*args, **kwargs)

        if "samples" in kwargs:
            return kwargs["samples"]
        elif "magnitude_spectrogram" in kwargs:
            return kwargs["magnitude_spectrogram"]
        else:
            return args[0]

