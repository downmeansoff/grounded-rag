"""Ответ с цитатами поверх найденных чанков. LLM — GigaChat.

Отвечает только по переданному контексту: если в найденных чанках нет
ответа, модель обязана явно сказать, что не нашла его, а не придумывать —
тот же принцип, что и `economics_unreliable` в tenderhunt: неизвестность
не превращается молча в правдоподобный факт.
"""

from __future__ import annotations

from grounded_rag.llm import GigaChatModel
from grounded_rag.store.postgres import SearchHit

SYSTEM_PROMPT = (
    "Ты отвечаешь на вопросы по тендерной документации. "
    "Используй только текст блоков ниже, помеченных [1], [2] и так далее. "
    "После каждого факта в ответе указывай номер источника в квадратных скобках. "
    'Если в блоках нет ответа на вопрос — прямо напиши "в найденных документах '
    'ответа нет", не добавляй факты из общих знаний и не угадывай.'
)


def _build_context(hits: list[SearchHit]) -> str:
    blocks = [
        f"[{i}] (тендер {hit.reg_number}, {hit.attachment_name}#{hit.chunk_index})\n{hit.text.strip()}"
        for i, hit in enumerate(hits, 1)
    ]
    return "\n\n".join(blocks)


def answer(query: str, hits: list[SearchHit], credentials: str, scope: str, model: str) -> str:
    if not hits:
        return "в найденных документах ответа нет"

    user_prompt = f"Вопрос: {query}\n\nДокументы:\n{_build_context(hits)}"

    with GigaChatModel(credentials, scope, model) as client:
        return client.complete(SYSTEM_PROMPT, user_prompt)
