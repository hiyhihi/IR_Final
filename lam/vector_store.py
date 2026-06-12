"""v5 store: vector + BM25 + hybrid retrieval, co persist/load VectorDB xuong disk."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

import config
from rag import bm25_rank, normalize_scores, rerank_results


def _rrf_fuse(rankings: list[list[tuple[int, float]]], rrf_k: int) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, (index, _score) in enumerate(ranking, start=1):
            scores[index] = scores.get(index, 0.0) + 1.0 / (rrf_k + rank)
    return scores


@dataclass
class HybridStore:
    chunks: list[str] = field(default_factory=list)
    embeddings: Optional[np.ndarray] = None
    doc_hash: str = ""

    def reset(self) -> None:
        self.chunks = []
        self.embeddings = None
        self.doc_hash = ""

    def add(self, chunks: list[str], embeddings: np.ndarray | None = None) -> None:
        self.chunks.extend(chunks)
        if embeddings is None or embeddings.size == 0:
            return
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be 2D")
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embs = (embeddings / norms).astype(np.float32)
        self.embeddings = embs if self.embeddings is None else np.vstack([self.embeddings, embs])

    # ----------------- Persist xuong disk -----------------

    def save(self, directory: str | Path) -> None:
        """Luu chunks + embeddings + meta xuong disk de load lai khi restart."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "backend": config.RETRIEVER_BACKEND,
            "embedding_model": config.EMBEDDING_MODEL,
            "doc_hash": self.doc_hash,
            "count": len(self.chunks),
            "has_embeddings": self.embeddings is not None,
        }
        (path / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        (path / "chunks.json").write_text(json.dumps(self.chunks, ensure_ascii=False), encoding="utf-8")
        emb_path = path / "embeddings.npy"
        if self.embeddings is not None:
            np.save(emb_path, self.embeddings)
        elif emb_path.exists():
            emb_path.unlink()

    def load(self, directory: str | Path) -> bool:
        """Load lai store tu disk. Tra ve True neu load thanh cong va khop cau hinh."""
        path = Path(directory)
        chunks_path = path / "chunks.json"
        meta_path = path / "meta.json"
        if not chunks_path.exists() or not meta_path.exists():
            return False
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        # Khong dung lai neu embedding model / backend da doi (vector khong tuong thich).
        if meta.get("embedding_model") != config.EMBEDDING_MODEL:
            return False
        if meta.get("backend") != config.RETRIEVER_BACKEND:
            return False
        try:
            chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        if not isinstance(chunks, list) or not chunks:
            return False

        embeddings = None
        emb_path = path / "embeddings.npy"
        if meta.get("has_embeddings") and emb_path.exists():
            try:
                embeddings = np.load(emb_path).astype(np.float32)
            except Exception:
                embeddings = None
            if embeddings is not None and len(embeddings) != len(chunks):
                return False

        self.chunks = [str(c) for c in chunks]
        self.embeddings = embeddings
        self.doc_hash = str(meta.get("doc_hash", ""))
        return True

    # ----------------- Tim kiem -----------------

    def _vector_search(self, query_emb: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        if self.embeddings is None or not self.chunks:
            return []
        q = query_emb.astype(np.float32).reshape(-1)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        scores = self.embeddings @ q
        k = min(top_k, len(self.chunks))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx]

    def search(self, query, top_k: int = 5) -> list[tuple[str, float]]:
        backend = config.RETRIEVER_BACKEND
        if not self.chunks:
            return []

        if backend in {"bm25", "tfidf"}:
            results = [(self.chunks[i], score) for i, score in bm25_rank(str(query), self.chunks, top_k)]
            return rerank_results(str(query), results)

        if backend in {"sbert", "openai", "vector"}:
            results = [(self.chunks[i], score) for i, score in self._vector_search(query, top_k)]
            return rerank_results("", results)

        if backend != "hybrid":
            raise RuntimeError(f"Unknown RETRIEVER_BACKEND={backend!r}")

        if not isinstance(query, tuple) or len(query) != 2:
            raise TypeError("HybridStore.search expects (question, query_embedding)")
        question, query_emb = query
        candidate_k = max(top_k, config.VECTOR_CANDIDATES)
        if config.RERANKER_MODEL:
            candidate_k = max(candidate_k, config.RERANK_CANDIDATES)

        # Neu chua co embeddings (vd backend hybrid nhung load tu disk thieu .npy),
        # fallback ve BM25 thuan de van tra loi duoc.
        if self.embeddings is None:
            results = [(self.chunks[i], score) for i, score in bm25_rank(question, self.chunks, top_k)]
            return rerank_results(question, results)

        vector_ranking = self._vector_search(query_emb, candidate_k)
        bm25_ranking = bm25_rank(question, self.chunks, candidate_k)

        if config.HYBRID_FUSION == "rrf":
            ranked = list(_rrf_fuse([vector_ranking, bm25_ranking], max(config.RRF_K, 1)).items())
        else:
            vector_scores = dict(vector_ranking)
            bm25_scores = dict(bm25_ranking)
            vector_norm = normalize_scores(vector_scores, higher_is_better=True)
            bm25_norm = normalize_scores(bm25_scores, higher_is_better=True)

            bm25_weight = min(max(config.BM25_WEIGHT, 0.0), 1.0)
            vector_weight = 1.0 - bm25_weight
            keys = set(vector_norm) | set(bm25_norm)
            ranked = [
                (index, vector_weight * vector_norm.get(index, 0.0) + bm25_weight * bm25_norm.get(index, 0.0))
                for index in keys
            ]

        ranked.sort(key=lambda item: item[1], reverse=True)
        results = [(self.chunks[index], score) for index, score in ranked[:candidate_k]]
        return rerank_results(question, results)[:top_k]

    def __len__(self) -> int:
        return len(self.chunks)


store = HybridStore()
