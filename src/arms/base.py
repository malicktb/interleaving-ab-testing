from abc import ABC, abstractmethod
import numpy as np


class BaseArm(ABC):

    def __init__(self, name):
        self.name = name
        self._is_trained = False

    @property
    def is_trained(self):
        return self._is_trained

    @abstractmethod
    def train(self, train_records):
        pass

    @abstractmethod
    def rank(self, record):
        pass

    def update(self, features, reward):
        pass
