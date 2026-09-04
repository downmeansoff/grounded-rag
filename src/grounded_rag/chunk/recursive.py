"""Рекурсивный чанкер: режет по абзацам, потом по предложениям, потом по символам.

Неделя 1 — простой baseline (recursive 512/overlap из архитектуры).
Contextual Retrieval (LLM-контекст перед каждым чанком) — отдельным шагом поверх этого,
после того как пайплайн ingestion → embed → store доказал себя на живом корпусе.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SEPARATORS = ["\n\n", "\n", ". ", " "]

DEFAULT_CHUNK_SIZE = 1500  # символов, ~ 400-500 токенов RU
DEFAULT_OVERLAP = 200


@dataclass
class Chunk:
    text: str
    index: int


def _split(text: str, separators: list[str]) -> list[str]:
    if not separators:
        return [text]
    sep, rest = separators[0], separators[1:]
    if sep not in text:
        return _split(text, rest)
    pieces = text.split(sep)
    return [p + sep for p in pieces[:-1]] + [pieces[-1]]


def _merge(pieces: list[str], size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if len(current) + len(piece) <= size:
            current += piece
            continue
        if current.strip():
            chunks.append(current.strip())
        tail = current[-overlap:] if overlap else ""
        current = tail + piece
        while len(current) > size:
            chunks.append(current[:size].strip())
            current = current[size - overlap :]
    if current.strip():
        chunks.append(current.strip())
    return chunks


def chunk_text(
    text: str, size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP
) -> list[Chunk]:
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []
    if len(text) <= size:
        return [Chunk(text, 0)]
    pieces = _split(text, _SEPARATORS)
    merged = _merge(pieces, size, overlap)
    return [Chunk(t, i) for i, t in enumerate(merged) if t]
