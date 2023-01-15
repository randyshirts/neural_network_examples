from abc import abstractmethod
from typing import Protocol

import numpy as np


class Model(Protocol):
    @abstractmethod
    def predict(
        self, X: np.array, y: np.array, parameters: dict[str, np.array]
    ) -> np.array:
        raise NotImplementedError

    @abstractmethod
    def model(
        self,
        X: np.array,
        Y: np.array,
        layers_dims: list[int],
        learning_rate: float = 0.0075,
        num_iterations: int = 3000,
        print_cost: bool = False,
    ) -> dict[str, float]:
        raise NotImplementedError
