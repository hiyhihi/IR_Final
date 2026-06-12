"""Shared text normalization helpers for retrieval and answer selection."""
from __future__ import annotations

import re
import unicodedata
from typing import Mapping

_OPTION_MARKER_RE = re.compile(r"(?:^|\n)\s*[ABCD][\).:\-]\s+", re.IGNORECASE)
_METADATA_PREFIXES = (
    "- id:",
    "- chu_de:",
    "- de_muc:",
    "- source_file:",
    "- source_info:",
    "- cross_refs:",
)


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    return text.replace("\u00a0", " ").replace("\ufeff", "")


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text or "")
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def normalize_legal_spacing(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<=\d)(?=[A-Za-zÀ-ỹ])", " ", text)
    text = re.sub(r"(?<=[A-Za-zÀ-ỹ])(?=\d)", " ", text)
    text = re.sub(r"\bm\s+2\b", "m2", text, flags=re.IGNORECASE)
    text = re.sub(r"\bn\s*/\s*a\b", "n/a", text, flags=re.IGNORECASE)
    return text


def clean_inline_text(text: str) -> str:
    text = normalize_legal_spacing(normalize_unicode(text))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_document_text(text: str) -> str:
    text = clean_inline_text(text)
    lines: list[str] = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            lines.append("")
            continue
        if line.startswith(_METADATA_PREFIXES):
            continue
        if line.startswith("## "):
            lines.extend(["", line, ""])
            continue
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def canonicalize_for_match(text: str) -> str:
    text = clean_inline_text(text).lower()
    text = strip_accents(text)
    text = re.sub(r"[^\w\s/%.-]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_for_search(text: str) -> list[str]:
    canonical = canonicalize_for_match(text)
    words = re.findall(r"[\w/%.-]+", canonical, flags=re.UNICODE)
    bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:])]
    trigrams = [f"{a}_{b}_{c}" for a, b, c in zip(words, words[1:], words[2:])]
    return words + bigrams + trigrams


def question_stem(question: str) -> str:
    question = clean_inline_text(question)
    parts = _OPTION_MARKER_RE.split(question, maxsplit=1)
    return parts[0].strip() if parts else question.strip()


def compose_retrieval_query(question: str, options: Mapping[str, str] | None = None, include_options: bool = False) -> str:
    stem = question_stem(question)
    if not include_options or not options:
        return stem
    option_lines = [f"{key}. {clean_inline_text(value)}" for key, value in sorted(options.items()) if key in "ABCD"]
    return stem + "\n" + "\n".join(option_lines) if option_lines else stem

