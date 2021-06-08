from abc import ABC, abstractmethod
from typing import List, Tuple

from dependencies_graph.data_types import StepsDependencies


class BaseStepsDependenciesExtractor(ABC):
    @abstractmethod
    def extract(self, question_id: str, question: str, decomposition:str, operators: List[str] = None,
                debug: dict = None) -> StepsDependencies:
        raise NotImplementedError()