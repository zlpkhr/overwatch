import abc
from threading import RLock

__all__ = ["Persistence", "MemoryPersistence"]


class Persistence(abc.ABC):
    """Abstract mapping between engine internal ids and external frame ids."""

    @abc.abstractmethod
    def put(self, internal_id: str, external_id: str) -> None: ...

    @abc.abstractmethod
    def get(self, internal_id: str) -> str | None: ...

    @abc.abstractmethod
    def delete(self, internal_id: str) -> None: ...

    @abc.abstractmethod
    def all_items(self) -> dict[str, str]: ...


class MemoryPersistence(Persistence):
    """In-memory persistence (non-durable)."""

    def __init__(self):
        self._map: dict[str, str] = {}
        self._lock = RLock()

    def put(self, internal_id: str, external_id: str) -> None:
        with self._lock:
            self._map[internal_id] = external_id

    def get(self, internal_id: str) -> str | None:
        with self._lock:
            return self._map.get(internal_id)

    def delete(self, internal_id: str) -> None:
        with self._lock:
            self._map.pop(internal_id, None)

    def all_items(self) -> dict[str, str]:
        with self._lock:
            return dict(self._map)
