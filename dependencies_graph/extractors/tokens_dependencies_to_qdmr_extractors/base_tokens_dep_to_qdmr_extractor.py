from abc import ABC, abstractmethod
from typing import List, Tuple

from evaluation.decomposition import Decomposition
from dependencies_graph.data_types import TokensDependencies


class BaseTokensDependenciesToQDMRExtractor(ABC):
    @abstractmethod
    def extract(self, tokens_dependencies: TokensDependencies, debug: dict = None) -> Decomposition:
        raise NotImplementedError()