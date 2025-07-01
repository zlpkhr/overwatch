import abc
from typing import Dict, List, Tuple


class IndexPlugin(abc.ABC):
    """Abstract base class for frame indexing plugins."""

    name: str  # unique identifier string (set by subclass)

    @abc.abstractmethod
    def index(self, internal_id: str, image: bytes) -> None:
        """Process raw image bytes and store plugin-specific data."""

    @abc.abstractmethod
    def score(self, query: str, top_k: int) -> List[Tuple[str, float, Dict]]:
        """Return list of (internal_id, probability 0-1, metadata)."""

    @abc.abstractmethod
    def delete(self, internal_id: str) -> None:
        """Remove all plugin data associated with a frame."""
