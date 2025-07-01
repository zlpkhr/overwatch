import abc
from typing import List


class QueryPlugin(abc.ABC):
    """Transforms a user query into one or more variants."""

    @abc.abstractmethod
    def expand(self, query: str) -> List[str]:
        """Return list of queries; must include the original."""
