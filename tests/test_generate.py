"""Генерация: честный отказ и формат контекста.

Тесты не ходят в сеть. Отказ при пустом retrieval — то место, где
«не знаю» не должно молча превратиться в правдоподобный ответ, поэтому
он проверяется без участия модели.
"""

from __future__ import annotations

from grounded_rag.generate.gigachat_llm import SYSTEM_PROMPT, _build_context, answer
from grounded_rag.store.postgres import SearchHit


def _hit(index: int = 0, text: str = "Работа с 8:00 до 18:00.") -> SearchHit:
    return SearchHit(
        reg_number="0312100006326000036",
        title="Оказание услуг по гардеробному обслуживанию",
        attachment_name="Описание объекта закупки",
        chunk_index=index,
        text=text,
        distance=0.2,
    )


def test_no_hits_gives_honest_refusal_without_calling_model():
    # Пустых credentials хватило бы для падения, если бы функция полезла в API.
    assert answer("какой график работы?", [], "", "", "") == "в найденных документах ответа нет"


def test_context_numbering_starts_at_one():
    context = _build_context([_hit(0), _hit(1)])
    assert context.startswith("[1] ")
    assert "\n\n[2] " in context


def test_context_carries_source_coordinates():
    context = _build_context([_hit(7)])
    assert "0312100006326000036" in context
    assert "Описание объекта закупки#7" in context


def test_context_keeps_chunk_text():
    context = _build_context([_hit(text="Круглосуточно, включая выходные.")])
    assert "Круглосуточно, включая выходные." in context


def test_system_prompt_demands_citations_and_refusal():
    assert "в найденных документах" in SYSTEM_PROMPT
    assert "квадратных скобках" in SYSTEM_PROMPT
