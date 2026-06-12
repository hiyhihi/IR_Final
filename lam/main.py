"""Student Server v2-based - FastAPI: /upload + /ask cho Teacher Server."""
from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

import lam.config as config
from lam.llm_client import ask_llm, compact_context, extract_options
from lam.rag import chunk_text
from lam.text_utils import clean_inline_text, compose_retrieval_query
from lam.vector_store import store

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("student-server")

app = FastAPI(title="Student RAG Server v5", version="5.0.0")

CURRENT_DOC_HASH = "no-doc"
ANSWER_CACHE: dict[str, dict] = {}


class UploadRequest(BaseModel):
    doc_id: Optional[str] = None
    text: str


class UploadResponse(BaseModel):
    status: str
    doc_id: Optional[str] = None
    chunks: int


class AskRequest(BaseModel):
    question: str
    options: Optional[dict[str, str] | list[str]] = None


class AskResponse(BaseModel):
    answer: str
    sources: List[str] = []


@app.on_event("startup")
def _startup() -> None:
    """Load lai VectorDB tu disk khi restart (spec BAT BUOC)."""
    global CURRENT_DOC_HASH
    if config.ENABLE_PERSIST and store.load(config.PERSIST_DIR):
        CURRENT_DOC_HASH = store.doc_hash or CURRENT_DOC_HASH
        log.info(
            "Loaded VectorDB tu disk: %d chunks (backend=%s, doc_hash=%s)",
            len(store), config.RETRIEVER_BACKEND, CURRENT_DOC_HASH,
        )
    else:
        log.info("Chua co VectorDB tren disk, cho Teacher goi /upload.")


@app.get("/")
def root():
    return {
        "status": "ok",
        "student_id": config.STUDENT_ID,
        "backend": config.RETRIEVER_BACKEND,
        "embedding_model": config.EMBEDDING_MODEL,
        "chunks": len(store),
        "doc_hash": CURRENT_DOC_HASH,
        "cached_answers": len(ANSWER_CACHE),
    }


def _index(chunks: list[str]) -> None:
    if config.RETRIEVER_BACKEND in {"tfidf", "bm25"}:
        store.add(chunks)
    else:
        from lam.rag import embed_texts

        embs = embed_texts(chunks)
        store.add(chunks, embs)


def _retrieve(question: str, options: dict[str, str], top_k: int) -> list[tuple[str, float]]:
    retrieval_query = compose_retrieval_query(
        question,
        options=options,
        include_options=config.RETRIEVAL_INCLUDE_OPTIONS,
    )
    if config.RETRIEVER_BACKEND in {"tfidf", "bm25"}:
        return store.search(retrieval_query, top_k=top_k)
    from lam.rag import embed_query

    q = embed_query(retrieval_query)
    if config.RETRIEVER_BACKEND == "hybrid":
        return store.search((retrieval_query, q), top_k=top_k)
    return store.search(q, top_k=top_k)


def _cache_dir() -> Path:
    path = Path(config.CACHE_DIR)
    path.mkdir(exist_ok=True)
    return path


def _question_key(question: str, options: dict[str, str]) -> str:
    payload = {
        "doc_hash": CURRENT_DOC_HASH,
        "question": " ".join(question.split()),
        "options": options,
        "backend": config.RETRIEVER_BACKEND,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_answer_cache() -> None:
    if ANSWER_CACHE or not config.ENABLE_ANSWER_CACHE:
        return
    path = _cache_dir() / "answer_cache.jsonl"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        key = row.get("key")
        answer = str(row.get("answer", "")).strip().upper()[:1]
        if key and answer in {"A", "B", "C", "D"}:
            row["answer"] = answer
            ANSWER_CACHE[key] = row


def _append_jsonl(filename: str, row: dict) -> None:
    path = _cache_dir() / filename
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _log_question(question: str, options: dict[str, str]) -> None:
    if not config.ENABLE_ANSWER_CACHE:
        return
    _append_jsonl(
        "questions_seen.jsonl",
        {"ts": time.time(), "doc_hash": CURRENT_DOC_HASH, "question": question, "options": options},
    )


def _get_cached_answer(question: str, options: dict[str, str]) -> dict | None:
    if not config.ENABLE_ANSWER_CACHE:
        return None
    _load_answer_cache()
    return ANSWER_CACHE.get(_question_key(question, options))


def _save_cached_answer(question: str, options: dict[str, str], answer: str, sources: list[str]) -> None:
    if not config.ENABLE_ANSWER_CACHE:
        return
    key = _question_key(question, options)
    row = {
        "ts": time.time(),
        "key": key,
        "doc_hash": CURRENT_DOC_HASH,
        "question": question,
        "options": options,
        "answer": answer,
        "sources": sources,
        "backend": config.RETRIEVER_BACKEND,
    }
    ANSWER_CACHE[key] = row
    _append_jsonl("answer_cache.jsonl", row)


# ----------------- Parse request linh hoat -----------------

async def _json_or_text(request: Request) -> dict:
    body = await request.body()
    if not body:
        return {}
    try:
        data = await request.json()
    except Exception:
        return {"text": body.decode("utf-8", errors="ignore")}
    return data if isinstance(data, dict) else {"text": str(data)}


def _first_text(data: dict, names: tuple[str, ...]) -> str:
    for name in names:
        value = data.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for value in data.values():
        if isinstance(value, dict):
            found = _first_text(value, names)
            if found:
                return found
    return ""


def _extract_doc_text(data: dict) -> str:
    text = _first_text(data, ("text", "content", "document", "doc", "raw_text", "body"))
    if text:
        return text
    docs = data.get("documents") or data.get("docs")
    if isinstance(docs, list):
        parts = []
        for item in docs:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(_extract_doc_text(item))
        return "\n\n".join(p for p in parts if p)
    return ""


def _normalize_options(raw) -> dict[str, str]:
    if isinstance(raw, dict):
        return {
            str(k).strip().upper()[:1]: clean_inline_text(str(v))
            for k, v in raw.items()
            if clean_inline_text(str(v))
        }
    if isinstance(raw, list):
        labels = "ABCD"
        return {labels[i]: clean_inline_text(str(v)) for i, v in enumerate(raw[:4]) if clean_inline_text(str(v))}
    return {}


def _extract_question(data: dict) -> tuple[str, dict[str, str]]:
    question = clean_inline_text(_first_text(data, ("question", "query", "prompt", "text", "content")))
    options = _normalize_options(data.get("options") or data.get("choices") or data.get("answers"))
    if options and not any(f"{k}." in question or f"{k})" in question for k in options):
        question = question + "\n" + "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
    if not options:
        options = extract_options(question)
    return question, options


# ----------------- Endpoints -----------------

@app.post("/upload", response_model=UploadResponse)
async def upload(request: Request) -> UploadResponse:
    global CURRENT_DOC_HASH
    data = await _json_or_text(request)
    doc_id = data.get("doc_id") or data.get("id") or data.get("document_id")
    text = _extract_doc_text(data)
    if not text:
        raise HTTPException(status_code=400, detail="empty text")

    CURRENT_DOC_HASH = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    # Reset store moi lan upload de khong lan du lieu giua cac vong.
    store.reset()
    store.doc_hash = CURRENT_DOC_HASH

    # Dump tai lieu de debug noi dung RAG.
    dump_dir = Path("uploads")
    dump_dir.mkdir(exist_ok=True)
    dump_path = dump_dir / f"{doc_id or 'doc'}_{uuid.uuid4().hex[:6]}.txt"
    dump_path.write_text(text, encoding="utf-8")
    log.info("Saved raw upload to %s (%d chars)", dump_path, len(text))

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="no chunk produced")

    log.info("Upload: %d chunks (backend=%s)", len(chunks), config.RETRIEVER_BACKEND)
    _index(chunks)

    # Persist VectorDB xuong disk de restart khong phai upload/embed lai.
    if config.ENABLE_PERSIST:
        store.save(config.PERSIST_DIR)
        log.info("Persisted VectorDB -> %s", config.PERSIST_DIR)

    doc_id = str(doc_id or f"doc_{uuid.uuid4().hex[:8]}")
    return UploadResponse(status="success", doc_id=doc_id, chunks=len(chunks))


@app.post("/ask", response_model=AskResponse)
async def ask(request: Request) -> AskResponse:
    data = await _json_or_text(request)
    question, options = _extract_question(data)
    if not question:
        raise HTTPException(status_code=400, detail="empty question")
    _log_question(question, options)

    cached = _get_cached_answer(question, options)
    if cached:
        answer = cached["answer"]
        sources = cached.get("sources") or []
        log.info("CACHE HIT Q=%s... -> %s", question[:60].replace("\n", " "), answer)
        return AskResponse(answer=answer, sources=sources)

    if len(store) == 0:
        log.warning("Khong co context, goi LLM khong context")
        answer = ask_llm(question, context="(khong co tai lieu)", options=options)
        _save_cached_answer(question, options, answer, [])
        return AskResponse(answer=answer, sources=[])

    hits = _retrieve(question, options, top_k=config.TOP_K)
    # Chi nhoi CONTEXT_TOP_K chunk tot nhat vao prompt de vua cua so ~4k token.
    top_sources = [c for c, _ in hits[: config.CONTEXT_TOP_K]]
    context = compact_context(top_sources)

    answer = ask_llm(question, context=context, options=options)
    sources = [c for c, _ in hits]
    _save_cached_answer(question, options, answer, top_sources)
    log.info("Q=%s... -> %s", question[:60].replace("\n", " "), answer)
    return AskResponse(answer=answer, sources=sources)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=config.HOST, port=config.PORT, reload=False)
