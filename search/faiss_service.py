import logging
import os
import threading

import faiss
import numpy as np

from ingest.models import Frame

logger = logging.getLogger(__name__)

class FaissIndexService:
    """
    Singleton service for managing a Faiss HNSW index of frame embeddings.
    Handles index building, loading, saving, adding embeddings, and searching.
    Thread-safe and auto-persistent.
    """
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.index = None  # Faiss index
        self.id_map = []   # List of frame IDs, index-aligned
        self.write_count = 0
        self.save_every = 10  # Save index every N writes
        self.index_path = 'media/faiss.index'
        self.id_map_path = 'media/faiss_id_map.npy'
        self._index_lock = threading.Lock()
        self._load_or_build()

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _load_or_build(self):
        if os.path.exists(self.index_path) and os.path.exists(self.id_map_path):
            try:
                self.load()
                logger.info("Loaded Faiss index and id_map from disk.")
                return
            except Exception as e:
                logger.warning(f"Failed to load Faiss index from disk: {e}. Rebuilding from DB.")
        self.build_from_db()

    def add_embedding(self, embedding, frame_id):
        with self._index_lock:
            emb = np.array(embedding, dtype=np.float32)
            if self.index is None:
                # Create new index if needed
                self.index = faiss.IndexHNSWFlat(emb.shape[0], 32)
                self.id_map = []
            self.index.add(emb.reshape(1, -1))
            self.id_map.append(frame_id)
            self.write_count += 1
            if self.write_count % self.save_every == 0:
                self.save()

    def search(self, query_embedding, k):
        with self._index_lock:
            if self.index is None or len(self.id_map) == 0:
                return []
            emb = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
            D, I = self.index.search(emb, k)
            results = []
            for idx, dist in zip(I[0], D[0]):
                if idx < len(self.id_map):
                    results.append((self.id_map[idx], float(dist)))
            return results

    def save(self):
        with self._index_lock:
            if self.index is not None:
                faiss.write_index(self.index, self.index_path)
                np.save(self.id_map_path, np.array(self.id_map, dtype=np.int64))
                logger.info("Faiss index and id_map saved to disk.")

    def load(self):
        with self._index_lock:
            self.index = faiss.read_index(self.index_path)
            self.id_map = np.load(self.id_map_path).tolist()

    def build_from_db(self):
        logger.info("Building Faiss index from DB...")
        frames = Frame.objects.exclude(embeddings=[])
        embeddings = []
        id_map = []
        for frame in frames:
            emb = np.array(frame.embeddings, dtype=np.float32)
            if emb.size > 0:
                embeddings.append(emb)
                id_map.append(frame.id)
        if embeddings:
            arr = np.stack(embeddings)
            self.index = faiss.IndexHNSWFlat(arr.shape[1], 32)
            self.index.add(arr)
            self.id_map = id_map
            logger.info(f"Built Faiss index with {len(id_map)} vectors.")
            self.save()
        else:
            self.index = None
            self.id_map = []
            logger.info("No embeddings found in DB. Faiss index is empty.") 