import uuid
from collections import defaultdict
from math import prod

from .persistence import MemoryPersistence, Persistence
from .plugins import IndexPlugin, QueryPlugin


class MatcherEngine:
    """Core orchestrator – indexes frames and performs text searches."""

    def __init__(
        self,
        *,
        index_plugins: dict[str, IndexPlugin] | None = None,
        query_transformer: QueryPlugin | None = None,
        persistence: Persistence | None = None,
    ) -> None:
        if not index_plugins:
            raise ValueError("At least one index plugin must be supplied")
        self._index_plugins = index_plugins
        self._query_transformer = query_transformer
        self._persist = persistence or MemoryPersistence()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def index(self, image: bytes, frame_id: str) -> None:
        """Ingest a single frame.

        Parameters
        ----------
        image : bytes
            Raw image bytes (JPEG/PNG).
        frame_id : str
            External caller-supplied identifier.
        """
        internal_id = uuid.uuid4().hex
        self._persist.put(internal_id, frame_id)
        for plugin in self._index_plugins.values():
            try:
                plugin.index(internal_id, image)
            except Exception:
                # swallow individual plugin failure to keep others alive
                continue

    def delete(self, frame_id: str) -> None:
        """Remove a frame across all plugins and persistence."""
        internal_ids = [
            k for k, v in self._persist.all_items().items() if v == frame_id
        ]
        for iid in internal_ids:
            self._persist.delete(iid)
            for plugin in self._index_plugins.values():
                try:
                    plugin.delete(iid)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(self, queries: str | list[str], top_k: int = 10) -> list[dict]:
        """Search for one or multiple query strings (text only)."""

        if isinstance(queries, str):
            query_list = [queries]
        else:
            query_list = list(queries)

        # apply optional transformer to entire list
        if self._query_transformer:
            try:
                query_list = self._query_transformer.expand(query_list)
            except Exception:
                pass

        queries_set = set(q for q in query_list if q)

        # collect scores --------------------------------------------------
        hits: dict[str, list[tuple[str, float, dict]]] = defaultdict(list)
        for q in queries_set:
            for name, plugin in self._index_plugins.items():
                try:
                    results = plugin.score(q, top_k)
                except Exception:
                    continue
                for internal_id, prob, meta in results:
                    hits[internal_id].append((name, prob, meta))

        # fuse duplicates -----------------------------------------------
        fused: list[tuple[str, float, dict[str, dict]]] = []
        for internal_id, lst in hits.items():
            probs = [p for _n, p, _m in lst]
            combined_prob = 1 - prod([(1 - p) for p in probs])
            meta_ns: dict[str, dict] = {}
            for n, _p, m in lst:
                if m:
                    meta_ns[n] = m
            fused.append((internal_id, combined_prob, meta_ns))

        fused.sort(key=lambda x: x[1], reverse=True)
        fused = fused[:top_k]

        # map back to external ids --------------------------------------
        response: list[dict] = []
        for internal_id, prob, meta in fused:
            ext_id = self._persist.get(internal_id) or internal_id
            response.append(
                {
                    "id": ext_id,
                    "prob": prob,
                    "meta": meta,
                }
            )
        return response
