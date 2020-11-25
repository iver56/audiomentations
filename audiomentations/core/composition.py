import random


class Compose:
    def __init__(self, transforms, p=1.0, shuffle=False):
        self.transforms = transforms
        self.p = p
        self.shuffle = shuffle

        name_list = []
        for transform in self.transforms:
            name_list.append(type(transform).__name__)
        self.__name__ = "_".join(name_list)

    def __call__(self, samples, sample_rate):
        transforms = self.transforms.copy()
        if random.random() < self.p:
            if self.shuffle:
                random.shuffle(transforms)
            for transform in transforms:
                samples = transform(samples, sample_rate)

        return samples

    def randomize_parameters(self, samples, sample_rate):
        """
        Randomize and define parameters of every transform in composition.
        """
        for transform in self.transforms:
            transform.randomize_parameters(samples, sample_rate)  

    def freeze_parameters(self):
        """
        Mark all parameters as frozen, i.e. do not randomize them for each call. This can be
        useful if you want to apply an effect chain with the exact same parameters to multiple
        sounds.
        """
        for transform in self.transforms:
            transform.freeze_parameters()

    def unfreeze_parameters(self):
        """
        Unmark all parameters as frozen, i.e. let them be randomized for each call.
        """
        for transform in self.transforms:
            transform.unfreeze_parameters()
