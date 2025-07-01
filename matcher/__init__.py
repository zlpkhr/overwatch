"""Matcher library – interfaces and memory persistence."""

from .persistence import MemoryPersistence, Persistence
from .plugins import IndexPlugin, QueryPlugin

__all__ = [
    "IndexPlugin",
    "QueryPlugin",
    "Persistence",
    "MemoryPersistence",
]
