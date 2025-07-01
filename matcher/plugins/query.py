import abc


class QueryPlugin(abc.ABC):
    """Transforms a user query into one or more variants."""

    @abc.abstractmethod
    def expand(self, queries: list[str]) -> list[str]:
        """Return expanded list of queries (must include originals)."""
