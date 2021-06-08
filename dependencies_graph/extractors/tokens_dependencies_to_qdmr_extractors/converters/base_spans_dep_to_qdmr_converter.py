from abc import ABC, abstractmethod

import networkx as nx

from dependencies_graph.data_types import SpansDependencies
from evaluation.decomposition import Decomposition


class BaseSpansDepToQdmrConverter(ABC):
    @abstractmethod
    def convert(self, spans_dependencies: SpansDependencies) -> Decomposition:
        raise NotImplementedError()
