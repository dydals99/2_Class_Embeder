"""ChromaDB vector store (CPU-friendly, local persistence)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import chromadb

from config import Config, get_config
from core.gemini import GeminiService
from core.schemas import ChunkRecord

if TYPE_CHECKING:
    from chromadb.api.models.Collection import Collection


class VectorStore:
    def __init__(
        self,
        collection_name: str,
        cfg: Config | None = None,
        gemini: GeminiService | None = None,
    ) -> None:
        self.cfg = cfg or get_config()
        self.gemini = gemini or GeminiService(self.cfg)
        self.client = chromadb.PersistentClient(path=str(self.cfg.chroma_dir))
        self.collection_name = collection_name
        self._collection: Collection | None = None

    @property
    def collection(self) -> "Collection":
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def reset_collection(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = None

    def add_chunks(self, chunks: list[ChunkRecord]) -> int:
        if not chunks:
            return 0
        ids = []
        documents = []
        metadatas = []
        embeddings = []
        for ch in chunks:
            ids.append(ch.chunk_id)
            documents.append(ch.text)
            meta = {k: str(v) for k, v in ch.metadata.items()}
            meta["doc_id"] = ch.doc_id
            metadatas.append(meta)
            embeddings.append(self.gemini.embed(ch.text))

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return len(chunks)

    def query(self, query_text: str, top_k: int | None = None) -> list[dict]:
        k = top_k or self.cfg.rag_top_k
        q_emb = self.gemini.embed(query_text)
        result = self.collection.query(
            query_embeddings=[q_emb],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        docs = result.get("documents") or [[]]
        metas = result.get("metadatas") or [[]]
        dists = result.get("distances") or [[]]
        for doc, meta, dist in zip(docs[0], metas[0], dists[0]):
            hits.append(
                {
                    "text": doc,
                    "metadata": meta,
                    "distance": dist,
                }
            )
        return hits
