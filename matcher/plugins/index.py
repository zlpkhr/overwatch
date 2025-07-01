import abc


class IndexPlugin(abc.ABC):
    """Abstract base class for frame indexing plugins."""

    @abc.abstractmethod
    def index(self, internal_id: str, image: bytes) -> None:
        """Process raw image bytes and store plugin-specific data."""

    @abc.abstractmethod
    def score(self, query: str, top_k: int) -> list[tuple[str, float, dict]]:
        """Return list of (internal_id, probability 0-1, metadata)."""

    @abc.abstractmethod
    def delete(self, internal_id: str) -> None:
        """Remove all plugin data associated with a frame."""
