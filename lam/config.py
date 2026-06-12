"""Cau hinh v2-based: clean data + prompt/postprocess tot hon."""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def _get(name: str, default: str | None = None, required: bool = False) -> str:
    val = os.getenv(name, default)
    if required and (val is None or val == ""):
        raise RuntimeError(f"Missing required env var: {name}")
    return val  # type: ignore[return-value]


def _flag(name: str, default: str = "true") -> bool:
    return _get(name, default).strip().lower() in {"1", "true", "yes", "on"}


# Teacher / Proxy
TEACHER_BASE_URL = _get("TEACHER_BASE_URL", "http://192.168.50.218:8000/api/v1").rstrip("/")
LLM_BASE_URL = _get("LLM_BASE_URL", f"{TEACHER_BASE_URL}/proxy").rstrip("/")

# Student
STUDENT_ID = _get("STUDENT_ID", required=True).strip().upper()
STUDENT_SERVER_URL = _get("STUDENT_SERVER_URL", required=True).rstrip("/")
if not STUDENT_SERVER_URL.startswith(("http://", "https://")):
    raise RuntimeError(
        f"STUDENT_SERVER_URL phai bat dau bang http:// hoac https:// "
        f"(dang la: {STUDENT_SERVER_URL!r})"
    )

# FastAPI
HOST = _get("HOST", "0.0.0.0")
PORT = int(_get("PORT", "5004"))

# LLM
LLM_MODEL = _get("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(_get("LLM_TEMPERATURE", "0.0"))
LLM_TIMEOUT = float(_get("LLM_TIMEOUT", "45"))

# Embedding / Retrieval
RETRIEVER_BACKEND = _get("RETRIEVER_BACKEND", "bm25").strip().lower()
OPENAI_EMBEDDING_MODEL = _get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_MODEL = _get("EMBEDDING_MODEL", "keepitreal/vietnamese-sbert")
DEVICE = _get("DEVICE", "cpu")

# Hybrid retrieval
VECTOR_CANDIDATES = int(_get("VECTOR_CANDIDATES", "36"))
BM25_WEIGHT = float(_get("BM25_WEIGHT", "0.55"))
HYBRID_FUSION = _get("HYBRID_FUSION", "weighted").strip().lower()
RRF_K = int(_get("RRF_K", "60"))
RERANKER_MODEL = _get("RERANKER_MODEL", "").strip()
RERANK_CANDIDATES = int(_get("RERANK_CANDIDATES", "24"))
RERANKER_BATCH_SIZE = int(_get("RERANKER_BATCH_SIZE", "4"))
RERANKER_MAX_LENGTH = int(_get("RERANKER_MAX_LENGTH", "512"))
RETRIEVAL_INCLUDE_OPTIONS = _flag("RETRIEVAL_INCLUDE_OPTIONS", "false")

# RAG / Chunking
CHUNK_SIZE = int(_get("CHUNK_SIZE", "820"))
CHUNK_OVERLAP = int(_get("CHUNK_OVERLAP", "140"))
# So chunk lay ra khi retrieve (de rerank chon loc).
TOP_K = int(_get("TOP_K", "10"))
# So chunk THUC SU nhoi vao prompt LLM. Giu nho de vua cua so ~4k token.
CONTEXT_TOP_K = int(_get("CONTEXT_TOP_K", "5"))
# Ngan sach ky tu context. ~2800 char tieng Viet ~ 2000-2400 token,
# chua het 4k de con cho cho cau hoi + options + system prompt.
MAX_CONTEXT_CHARS = int(_get("MAX_CONTEXT_CHARS", "3400"))

# Prompt / postprocess
PROMPT_MODE = _get("PROMPT_MODE", "evidence").strip().lower()
OPTION_SCORE_THRESHOLD = float(_get("OPTION_SCORE_THRESHOLD", "6.0"))
OPTION_SCORE_MARGIN = float(_get("OPTION_SCORE_MARGIN", "1.5"))
OVERRIDE_WITH_HEURISTIC = _flag("OVERRIDE_WITH_HEURISTIC", "false")

# Persist VectorDB xuong disk + answer cache.
PERSIST_DIR = _get("PERSIST_DIR", "vectordb")
ENABLE_PERSIST = _flag("ENABLE_PERSIST", "true")
CACHE_DIR = _get("CACHE_DIR", "cache")
ENABLE_ANSWER_CACHE = _flag("ENABLE_ANSWER_CACHE", "true")
