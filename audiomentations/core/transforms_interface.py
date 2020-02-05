import random
import uuid


class BasicTransform:
    def __init__(self, p=0.5):
        assert 0 <= p <= 1
        self.p = p
        self.parameters = {"should_apply": None}
        self.input_sample_rate = None
        self.id = "{}_{}".format(self.__class__.__name__, uuid.uuid4())

    def __call__(self, samples, sample_rate):
        self.sample_rate = sample_rate
        self.randomize_parameters()
        if random.random() < self.p:
            return self.apply(samples, sample_rate)
        return samples

    def randomize_parameters(self):
        self.parameters["should_apply"] = random.random() < self.p

    def apply(self, samples, sample_rate):
        raise NotImplementedError
