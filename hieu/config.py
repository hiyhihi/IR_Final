"""Cau hinh v5: hybrid RAG + cache + persist VectorDB, toi uu cho LLM proxy ~4k context."""
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

# LLM. Teacher doi toi da 60s/cau -> (LLM_RETRIES+1) * LLM_TIMEOUT phai < 60.
LLM_MODEL = _get("LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(_get("LLM_TEMPERATURE", "0.0"))
LLM_TIMEOUT = float(_get("LLM_TIMEOUT", "25"))
LLM_RETRIES = int(_get("LLM_RETRIES", "1"))

# Embedding / Retrieval. Spec BAT BUOC: keepitreal/vietnamese-sbert.
RETRIEVER_BACKEND = _get("RETRIEVER_BACKEND", "hybrid").strip().lower()
OPENAI_EMBEDDING_MODEL = _get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_MODEL = _get("EMBEDDING_MODEL", "keepitreal/vietnamese-sbert")
DEVICE = _get("DEVICE", "cpu")


def _hf_model_cached(model: str) -> bool:
    from pathlib import Path

    hub = os.getenv("HF_HUB_CACHE")
    if not hub:
        home = os.getenv("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        hub = os.path.join(home, "hub")
    return (Path(hub) / ("models--" + model.replace("/", "--"))).is_dir()


# Model da nam trong cache -> bat che do offline TRUOC khi import transformers,
# chan moi network check cua HF (khi thi offline cac check nay treo cho timeout).
if _flag("HF_OFFLINE_AUTO", "true") and _hf_model_cached(EMBEDDING_MODEL):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Hybrid retrieval
VECTOR_CANDIDATES = int(_get("VECTOR_CANDIDATES", "30"))
BM25_WEIGHT = float(_get("BM25_WEIGHT", "0.6"))
# Cach tron diem hybrid: "weighted" (chuan hoa + trong so) hoac "rrf".
HYBRID_FUSION = _get("HYBRID_FUSION", "weighted").strip().lower()
RRF_K = int(_get("RRF_K", "60"))
# Query retrieval mac dinh chi dung than cau hoi, khong nhoi options.
RETRIEVAL_INCLUDE_OPTIONS = _flag("RETRIEVAL_INCLUDE_OPTIONS", "false")
RERANKER_MODEL = _get("RERANKER_MODEL", "").strip()
RERANK_CANDIDATES = int(_get("RERANK_CANDIDATES", "20"))
RERANKER_BATCH_SIZE = int(_get("RERANKER_BATCH_SIZE", "4"))
RERANKER_MAX_LENGTH = int(_get("RERANKER_MAX_LENGTH", "512"))

# RAG / Chunking
CHUNK_SIZE = int(_get("CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(_get("CHUNK_OVERLAP", "120"))
# Bo manh vun qua ngan (thuong la tieu de/ky tu thua), khong dang luu vao index.
MIN_CHUNK_LEN = int(_get("MIN_CHUNK_LEN", "40"))
# Bo chunk gan trung lap khi retrieve (cosine > nguong). Dat >=1.0 de tat.
DEDUP_THRESHOLD = float(_get("DEDUP_THRESHOLD", "0.92"))
# So chunk lay ra khi retrieve (de rerank chon loc).
TOP_K = int(_get("TOP_K", "8"))
# So chunk THUC SU nhoi vao prompt LLM. Giu nho de vua cua so ~4k token.
CONTEXT_TOP_K = int(_get("CONTEXT_TOP_K", "4"))
# Ngan sach ky tu context. ~2800 char tieng Viet ~ 2000-2400 token,
# chua het 4k de con cho cho cau hoi + options + system prompt.
MAX_CONTEXT_CHARS = int(_get("MAX_CONTEXT_CHARS", "2800"))

# Prompt / postprocess
# "evidence": prompt co huong dan doi chieu bang chung; "compact": prompt gon.
PROMPT_MODE = _get("PROMPT_MODE", "evidence").strip().lower()
# Heuristic de LLM khi diem bang chung vuot nguong + cach biet du lon (tat mac dinh).
OPTION_SCORE_THRESHOLD = float(_get("OPTION_SCORE_THRESHOLD", "6.0"))
OPTION_SCORE_MARGIN = float(_get("OPTION_SCORE_MARGIN", "1.5"))
OVERRIDE_WITH_HEURISTIC = _flag("OVERRIDE_WITH_HEURISTIC", "false")

# Persist VectorDB xuong disk (spec BAT BUOC) + answer cache.
PERSIST_DIR = _get("PERSIST_DIR", "vectordb")
ENABLE_PERSIST = _flag("ENABLE_PERSIST", "true")
CACHE_DIR = _get("CACHE_DIR", "cache")
ENABLE_ANSWER_CACHE = _flag("ENABLE_ANSWER_CACHE", "true")
