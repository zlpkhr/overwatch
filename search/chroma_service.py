import os

import chromadb


class ChromaService:
    _client = None
    _collection = None
    _persist_path = os.path.join("media", "chroma_data")

    @classmethod
    def get_collection(cls):
        if cls._client is None:
            os.makedirs(cls._persist_path, exist_ok=True)
            cls._client = chromadb.PersistentClient(path=cls._persist_path)
        if cls._collection is None:
            cls._collection = cls._client.get_or_create_collection(
                name="frames",
                metadata={"hnsw:space": "cosine"}
            )
        return cls._collection
