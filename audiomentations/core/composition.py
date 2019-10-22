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
