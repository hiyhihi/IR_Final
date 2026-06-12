"""v4 RAG helpers: chunking, embeddings, BM25, hybrid retrieval, optional rerank."""
from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache
from typing import Any

import numpy as np

import config
from text_utils import clean_document_text, tokenize_for_search

_RERANKER_CACHE: dict[tuple[str, str], Any] = {}


def clean_text(text: str) -> str:
    # NFKC + sua dinh chu-so + loc dong metadata (- id:, - chu_de:, ...) cua tai lieu.
    return clean_document_text(text)


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…;])\s+")


def _split_sentences(paragraph: str) -> list[str]:
    sentences: list[str] = []
    for part in _SENTENCE_SPLIT_RE.split(paragraph):
        sentences.extend(s.strip() for s in part.split("\n"))
    return [s for s in sentences if s]


def _overlap_tail(text: str, overlap: int) -> str:
    """Lay duoi `overlap` ky tu, ne cat giua tu."""
    if overlap <= 0:
        return ""
    if len(text) <= overlap:
        return text
    tail = text[-overlap:]
    idx = tail.find(" ")
    return tail[idx + 1 :] if idx != -1 else tail


def _hard_split(sentence: str, chunk_size: int, overlap: int) -> list[str]:
    pieces: list[str] = []
    start = 0
    step = max(chunk_size - overlap, 1)
    while start < len(sentence):
        pieces.append(sentence[start : start + chunk_size])
        if start + chunk_size >= len(sentence):
            break
        start += step
    return pieces


def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    """Tach cau roi gop greedy thanh chunk <= chunk_size, overlap bam ranh gioi tu."""
    chunk_size = chunk_size or config.CHUNK_SIZE
    overlap = overlap if overlap is not None else config.CHUNK_OVERLAP
    text = clean_text(text)
    if not text:
        return []

    sentences: list[str] = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if paragraph:
            sentences.extend(_split_sentences(paragraph))

    chunks: list[str] = []
    parts: list[str] = []  # cau dang gop; phan tu dau co the la duoi overlap cua chunk truoc
    length = 0
    has_new = False  # parts co noi dung moi ngoai duoi overlap khong

    def seed_overlap(next_len: int) -> None:
        """Seed duoi overlap cho chunk ke tiep, chi khi con cho cho cau sap toi."""
        nonlocal parts, length, has_new
        tail = _overlap_tail(chunks[-1], overlap) if chunks else ""
        if tail and len(tail) + 1 + next_len <= chunk_size:
            parts, length = [tail], len(tail)
        else:
            parts, length = [], 0
        has_new = False

    for sentence in sentences:
        if len(sentence) > chunk_size:
            if has_new:
                chunks.append(" ".join(parts))
            chunks.extend(_hard_split(sentence, chunk_size, overlap))
            seed_overlap(0)
            continue
        needed = len(sentence) + (1 if parts else 0)
        if length + needed > chunk_size and parts:
            if has_new:
                chunks.append(" ".join(parts))
                seed_overlap(len(sentence))
            else:
                # parts chi co duoi overlap ma cong cau moi van vuot -> bo duoi
                parts, length = [], 0
            needed = len(sentence) + (1 if parts else 0)
        parts.append(sentence)
        length += needed
        has_new = True

    if has_new:
        chunks.append(" ".join(parts))

    # Manh ngan hon MIN_CHUNK_LEN: gop vao chunk truoc/sau neu vua, KHONG vut
    # (co the la tieu de chua tu khoa quan trong).
    deduped: list[str] = []
    for chunk in chunks:
        if deduped and deduped[-1] == chunk:
            continue
        if (
            len(chunk) < config.MIN_CHUNK_LEN
            and deduped
            and len(deduped[-1]) + 1 + len(chunk) <= chunk_size
        ):
            deduped[-1] = deduped[-1] + " " + chunk
            continue
        deduped.append(chunk)
    if len(deduped) > 1 and len(deduped[0]) < config.MIN_CHUNK_LEN:
        if len(deduped[0]) + 1 + len(deduped[1]) <= chunk_size:
            deduped[1] = deduped[0] + " " + deduped[1]
            deduped.pop(0)
    return deduped


@lru_cache(maxsize=1)
def _load_sbert():
    from sentence_transformers import SentenceTransformer

    # Uu tien cache local: thi offline khong cho HF check mang (timeout cham).
    try:
        return SentenceTransformer(
            config.EMBEDDING_MODEL, device=config.DEVICE, local_files_only=True
        )
    except Exception:
        return SentenceTransformer(config.EMBEDDING_MODEL, device=config.DEVICE)


def _sbert_encode(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    model = _load_sbert()
    embs = model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=False,
    )
    return embs.astype(np.float32)


def embed_texts(texts: list[str]) -> np.ndarray:
    backend = config.RETRIEVER_BACKEND
    if backend == "openai":
        from llm_client import embed_openai

        return embed_openai(texts)
    if backend in {"sbert", "hybrid", "vector"}:
        return _sbert_encode(texts)
    if backend == "bm25":
        return np.zeros((len(texts), 0), dtype=np.float32)
    raise RuntimeError(f"embed_texts not supported for RETRIEVER_BACKEND={backend!r}")


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]


def tokenize_text(text: str) -> list[str]:
    # Bo dau + bigram + trigram: BM25 van khop khi cau hoi go thieu/khac dau.
    return tokenize_for_search(text)


class BM25Index:
    """BM25 voi posting list, dung san khi index — moi query khong tokenize lai corpus."""

    K1 = 1.5
    B = 0.75

    def __init__(self, chunks: list[str]) -> None:
        tokenized = [tokenize_text(chunk) for chunk in chunks]
        self.doc_count = len(tokenized)
        self.doc_lens = [len(tokens) for tokens in tokenized]
        self.avg_len = sum(self.doc_lens) / max(self.doc_count, 1)
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for index, tokens in enumerate(tokenized):
            for term, freq in Counter(tokens).items():
                self.postings.setdefault(term, []).append((index, freq))

    def rank(self, question: str, k: int) -> list[tuple[int, float]]:
        query_terms = tokenize_text(question)
        if not query_terms or not self.doc_count:
            return []
        avg = max(self.avg_len, 1.0)
        scores: dict[int, float] = {}
        for term in query_terms:
            plist = self.postings.get(term)
            if not plist:
                continue
            df = len(plist)
            idf = math.log(1 + (self.doc_count - df + 0.5) / (df + 0.5))
            for index, freq in plist:
                denom = freq + self.K1 * (1 - self.B + self.B * self.doc_lens[index] / avg)
                scores[index] = scores.get(index, 0.0) + idf * freq * (self.K1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked[:k]


def bm25_rank(question: str, chunks: list[str], k: int) -> list[tuple[int, float]]:
    """Tien ich one-shot; voi store dung BM25Index dung san de tranh build lai moi query."""
    return BM25Index(chunks).rank(question, k)


def normalize_scores(scores: dict[int, float], higher_is_better: bool) -> dict[int, float]:
    if not scores:
        return {}
    values = list(scores.values())
    if not higher_is_better:
        values = [-value for value in values]
    min_value = min(values)
    max_value = max(values)
    if max_value == min_value:
        return {key: 1.0 for key in scores}
    normalized: dict[int, float] = {}
    for key, value in scores.items():
        comparable = value if higher_is_better else -value
        normalized[key] = (comparable - min_value) / (max_value - min_value)
    return normalized


def load_reranker(model_name: str, device: str):
    cache_key = (model_name, device)
    if cache_key in _RERANKER_CACHE:
        return _RERANKER_CACHE[cache_key]

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    reranker = {"torch": torch, "tokenizer": tokenizer, "model": model, "device": device}
    _RERANKER_CACHE[cache_key] = reranker
    return reranker


def rerank_results(question: str, results: list[tuple[str, float]]) -> list[tuple[str, float]]:
    model_name = config.RERANKER_MODEL
    if not model_name or not results:
        return results[: config.TOP_K]

    reranker = load_reranker(model_name, config.DEVICE)
    torch = reranker["torch"]
    tokenizer = reranker["tokenizer"]
    model = reranker["model"]
    device = reranker["device"]

    scored: list[tuple[str, float]] = []
    for start in range(0, len(results), config.RERANKER_BATCH_SIZE):
        batch = results[start : start + config.RERANKER_BATCH_SIZE]
        pairs = [[question, chunk] for chunk, _score in batch]
        encoded = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=config.RERANKER_MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits
            scores = logits.view(-1).float().cpu().tolist()
        scored.extend((chunk, float(score)) for (chunk, _base_score), score in zip(batch, scores))

    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[: config.TOP_K]
