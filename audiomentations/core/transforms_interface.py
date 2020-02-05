import random
import uuid


class BasicTransform:
    def __init__(self, p=0.5):
        assert 0 <= p <= 1
        self.p = p
        self.parameters = {"should_apply": None}
        self.id = "{}_{}".format(self.__class__.__name__, uuid.uuid4())
        self.freeze_parameters = False

    def __call__(self, samples, sample_rate):
        if not self.freeze_parameters:
            self.randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            return self.apply(samples, sample_rate)
        return samples

    def randomize_parameters(self, samples, sample_rate):
        self.parameters["should_apply"] = random.random() < self.p

    def apply(self, samples, sample_rate):
        raise NotImplementedError
