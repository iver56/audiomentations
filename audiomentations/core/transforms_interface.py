import random


class BasicTransform:
    def __init__(self, p=0.5):
        assert 0 <= p <= 1
        self.p = p

    def __call__(self, samples, sample_rate):
        if random.random() < self.p:
            return self.apply(samples, sample_rate)
        return samples

    def apply(self, samples, sample_rate):
        raise NotImplementedError
