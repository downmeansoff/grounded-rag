"""Генерация ответа по найденным чанкам.

Промпт и подпись блока приходят из профиля предметной области: требование
отвечать только по блокам и прямо отказываться общее, а вот «тендерной
документации» против «документам корпуса» и «тендер 0312...» против
«документ readme» зависят от того, что индексируется.
"""

from __future__ import annotations

from grounded_rag.domain.base import DomainProfile
from grounded_rag.llm import GigaChatModel
from grounded_rag.store.postgres import SearchHit

REFUSAL = "в найденных документах ответа нет"


def build_context(profile: DomainProfile, hits: list[SearchHit]) -> str:
    blocks = [
        f"[{i}] ({profile.citation(hit.doc_id, hit.part_name, hit.chunk_index)})\n{hit.text.strip()}"
        for i, hit in enumerate(hits, 1)
    ]
    return "\n\n".join(blocks)


def answer(
    query: str,
    hits: list[SearchHit],
    profile: DomainProfile,
    credentials: str,
    scope: str,
    model: str,
) -> str:
    if not hits:
        return REFUSAL
    user_prompt = f"Вопрос: {query}\n\nДокументы:\n{build_context(profile, hits)}"
    with GigaChatModel(credentials, scope, model) as client:
        return client.complete(profile.answer_system, user_prompt)
