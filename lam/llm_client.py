"""LLM client with prompt, cleaning and post-processing tuned for MCQ legal QA."""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Mapping

import numpy as np
from openai import OpenAI

import config
from text_utils import canonicalize_for_match, clean_inline_text, question_stem, strip_accents

SYSTEM_PROMPT = (
    "Ban la tro ly tra loi trac nghiem tieng Viet ve van ban phap ly. "
    "Chi duoc dua vao CONTEXT. "
    "Uu tien chi tiet xuat hien truc tiep trong CONTEXT nhu co quan, muc, thoi han, con so, doi tuong, tham quyen. "
    "Neu cau hoi co tu phu dinh nhu KHONG, tru, khac, hay chon phuong an khong phu hop voi CONTEXT. "
    "Chi tra ve duy nhat mot chu cai in hoa A, B, C hoac D."
)

USER_PROMPT_EVIDENCE = """CONTEXT:
{context}

CAU HOI:
{question}

CAC LUA CHON:
{options_block}

Huong dan noi bo:
- Tim chi tiet trong CONTEXT khop nhat voi tung lua chon.
- So sanh cac chu the, tham quyen, thoi han, con so, dieu kien.
- Neu co phu dinh KHONG/tru/khac, dao nguoc tieu chi chon.

Tra loi chi mot chu cai A/B/C/D:"""

USER_PROMPT_COMPACT = """CONTEXT:
{context}

CAU HOI:
{question}
{options_block}
Tra loi chi mot chu cai A/B/C/D:"""

_WEAK_TOKENS = {
    "la", "là", "gi", "gì", "nao", "nào", "de", "để", "cua", "của", "tren", "trên",
    "trong", "mot", "một", "cac", "các", "theo", "quy", "dinh", "định", "quy định", "phap", "pháp",
    "luat", "luật", "thong", "thông", "tu", "tư", "nghi", "nghị", "dinh", "định", "noi", "nội",
}
_NEGATION_MARKERS = (" khong ", " không ", " tru ", " trừ ", " khac ", " khác ")
_LETTER_RE = re.compile(r"\b([ABCD])\b")
_OPTION_RE = re.compile(
    r"(?:^|\n|\s)([ABCD])[\).:\-]\s*(.*?)(?=(?:\n|\s)[ABCD][\).:\-]\s*|$)",
    re.IGNORECASE | re.DOTALL,
)


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI(
        base_url=config.LLM_BASE_URL,
        api_key=config.STUDENT_ID,
        timeout=config.LLM_TIMEOUT,
        max_retries=0,
    )


def parse_answer(text: str) -> str:
    if not text:
        return "A"
    s = text.strip().upper()
    if s in {"A", "B", "C", "D"}:
        return s
    match = _LETTER_RE.search(s)
    if match:
        return match.group(1)
    for ch in s:
        if ch in "ABCD":
            return ch
    return "A"


def extract_options(question: str) -> dict[str, str]:
    options: dict[str, str] = {}
    for letter, text in _OPTION_RE.findall(question or ""):
        letter = letter.upper()
        cleaned = clean_inline_text(text)
        if cleaned:
            options[letter] = cleaned
    return options


def _options_block(options: Mapping[str, str] | None) -> str:
    if not options:
        return ""
    return "\n".join(f"{key}. {clean_inline_text(value)}" for key, value in sorted(options.items()) if key in "ABCD")


def _informative_tokens(text: str) -> list[str]:
    raw = re.findall(r"[\wÀ-ỹ/%.-]+", strip_accents(text.lower()), flags=re.UNICODE)
    return [tok for tok in raw if len(tok) > 1 and tok not in _WEAK_TOKENS]


def _split_windows(context: str) -> list[str]:
    parts = [
        clean_inline_text(part)
        for part in re.split(r"\n---\n|\n\n+|(?<=[.!?])\s+", context or "")
        if clean_inline_text(part)
    ]
    return parts or [clean_inline_text(context or "")]


def _number_tokens(text: str) -> set[str]:
    return set(re.findall(r"\d+[\d./,%-]*", text))


def score_options(question: str, context: str, options: Mapping[str, str] | None) -> dict[str, float]:
    options = dict(options or {})
    if not options:
        return {}

    stem_tokens = set(_informative_tokens(question_stem(question)))
    windows = _split_windows(context)
    canonical_windows = [canonicalize_for_match(window) for window in windows]
    negated = any(marker in f" {canonicalize_for_match(question)} " for marker in _NEGATION_MARKERS)

    scored: dict[str, float] = {}
    for label, option_text in sorted(options.items()):
        option_clean = clean_inline_text(option_text)
        option_canon = canonicalize_for_match(option_clean)
        option_tokens = _informative_tokens(option_clean)
        option_numbers = _number_tokens(option_canon)
        best = 0.0

        for window, window_canon in zip(windows, canonical_windows):
            window_tokens = set(_informative_tokens(window))
            q_overlap = len(stem_tokens & window_tokens)
            opt_overlap = sum(1 for tok in option_tokens if tok in window_tokens)
            exact_hit = 1 if option_canon and option_canon in window_canon else 0
            number_hit = sum(1 for tok in option_numbers if tok in window_canon)
            all_numbers_hit = 1 if option_numbers and number_hit == len(option_numbers) else 0
            article_hit = 1 if any(tok.startswith("dieu") or tok.startswith("khoan") for tok in option_tokens if tok in window_tokens) else 0
            score = (
                exact_hit * 6.0
                + all_numbers_hit * 4.0
                + number_hit * 1.8
                + opt_overlap * 1.25
                + min(q_overlap, 6) * 0.35
                + article_hit * 0.75
            )
            if option_tokens and opt_overlap == len(option_tokens):
                score += 1.5
            best = max(best, score)

        scored[label] = -best if negated else best
    return scored


def heuristic_answer(question: str, context: str, options: Mapping[str, str] | None = None) -> str:
    options = dict(options or extract_options(question))
    if not options:
        return "A"

    scores = score_options(question, context, options)
    if scores:
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked[0][0]

    labels = sorted(k for k in options if k in "ABCD")
    return labels[0] if labels else "A"


def compact_context(chunks: list[str] | str, max_chars: int | None = None) -> str:
    max_chars = max_chars or config.MAX_CONTEXT_CHARS
    if isinstance(chunks, str):
        chunks = [chunks]

    picked: list[str] = []
    used = 0
    for chunk in chunks:
        cleaned = clean_inline_text(chunk)
        if not cleaned:
            continue
        remaining = max_chars - used
        if remaining <= 0:
            break
        if len(cleaned) > remaining:
            cleaned = cleaned[:remaining].rsplit(" ", 1)[0].strip() or cleaned[:remaining]
        if cleaned:
            picked.append(cleaned)
            used += len(cleaned) + 5
    return "\n---\n".join(picked)


def _build_user_prompt(question: str, context: str, options: Mapping[str, str] | None) -> str:
    template = USER_PROMPT_EVIDENCE if config.PROMPT_MODE == "evidence" else USER_PROMPT_COMPACT
    return template.format(
        context=context,
        question=clean_inline_text(question),
        options_block=_options_block(options),
    )


def _should_override(scores: dict[str, float], llm_answer: str) -> str | None:
    if not scores or not config.OVERRIDE_WITH_HEURISTIC:
        return None
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_label, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else float("-inf")
    if top_score >= config.OPTION_SCORE_THRESHOLD and (top_score - second_score) >= config.OPTION_SCORE_MARGIN:
        if llm_answer != top_label:
            return top_label
    return None


def ask_llm(question: str, context: str, options: Mapping[str, str] | None = None) -> str:
    options = dict(options or extract_options(question))
    scores = score_options(question, context, options)
    prompt = _build_user_prompt(question, context, options)
    try:
        response = _client().chat.completions.create(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=5,
        )
        raw = response.choices[0].message.content or ""
        parsed = parse_answer(raw)
        override = _should_override(scores, parsed)
        if override:
            return override
        if not raw.strip():
            return heuristic_answer(question, context, options)
        return parsed
    except Exception:
        return heuristic_answer(question, context, options)


def embed_openai(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    response = _client().embeddings.create(
        model=config.OPENAI_EMBEDDING_MODEL,
        input=texts,
    )
    return np.array([row.embedding for row in response.data], dtype=np.float32)
