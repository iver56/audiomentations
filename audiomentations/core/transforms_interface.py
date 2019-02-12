import random


class BasicTransform:
    def __init__(self,  p=0.5):
        self.p = p

    def __call__(self, samples, sample_rate):
        if random.random() < self.p:
            return self.apply(samples, sample_rate)

    def apply(self, samples, sample_rate):
        raise NotImplementedError
