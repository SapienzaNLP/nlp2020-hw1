import numpy as np
from typing import List, Tuple

from model import Model


def build_model(device: str) -> Model:
    # STUDENT: return StudentModel()
    # STUDENT: your model MUST be loaded on the device "device" indicates
    return RandomBaseline()


class RandomBaseline(Model):

    options = [
        ('LOC', 98412),
        ('O', 2512990),
        ('ORG', 71633),
        ('PER', 115758)
    ]

    def __init__(self):

        self._options = [option[0] for option in self.options]
        self._weights = np.array([option[1] for option in self.options])
        self._weights = self._weights / self._weights.sum()

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        return [[str(np.random.choice(self._options, 1, p=self._weights)[0]) for _x in x] for x in tokens]


class StudentModel(Model):
    
    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
        # STUDENT: implement here your predict function
        # remember to respect the same order of tokens!
        pass
