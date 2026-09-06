"""Contextual Retrieval: промпт, кэш и поведение при обрыве.

Модель не вызывается: проверяется не качество формулировок GigaChat, а то, что
в промпт кладётся нужное, что за один и тот же чанк не платят дважды и что
упавший вызов не роняет весь ingest.
"""

from __future__ import annotations

import json

from grounded_rag.contextual.cache import ContextCache, cache_key
from grounded_rag.contextual.contextualizer import Contextualizer, enrich
from grounded_rag.domain.base import Document, Part
from grounded_rag.domain.tenders import TendersProfile
from grounded_rag.llm import QuotaExhausted

# Формулировки промпта живут в профиле, поэтому без него контекстуализатор не
# собирается: он не знает, что индексируется, и это правильно.
PROFILE = TendersProfile()

DOC = Document(
    doc_id="0312100006326000036",
    title="Оказание услуг по гардеробному обслуживанию",
    source_path="/tmp/doc.txt",
    meta={"Заказчик": "ГБУК Музей", "НМЦК": "450000"},
    parts=[
        Part(
            name="Описание объекта закупки",
            ext="docx",
            text="Предметом закупки является гардеробное обслуживание посетителей музея. " * 40,
        )
    ],
)

CHUNK = "Оплата производится в течение 15 рабочих дней с даты подписания акта."


class FakeModel:
    """Считает вызовы и запоминает промпты; можно заставить падать."""

    def __init__(self, answer: str = "Раздел про порядок оплаты.", fail: bool = False) -> None:
        self.answer = answer
        self.fail = fail
        self.prompts: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.prompts.append((system, user))
        if self.fail:
            raise RuntimeError("429 too many requests")
        return self.answer


def test_enrich_puts_context_before_text():
    assert enrich("Про оплату.", "Текст.") == "Про оплату.\n\nТекст."


def test_enrich_without_context_returns_text_unchanged():
    # Чанк без контекста не должен получить пустую строку и два перевода строки
    # в начало: иначе эмбеддинги обогащённых и необогащённых чанков разойдутся.
    assert enrich("", "Текст.") == "Текст."
    assert enrich("   ", "Текст.") == "Текст."


def test_prompt_carries_metadata_head_and_chunk():
    model = FakeModel()
    Contextualizer(model, PROFILE).context_for(DOC, "Описание объекта закупки", CHUNK)

    _, user = model.prompts[0]
    assert "0312100006326000036" in user
    assert "Оказание услуг по гардеробному обслуживанию" in user
    assert "ГБУК Музей" in user
    assert "Описание объекта закупки" in user
    assert "гардеробное обслуживание посетителей музея" in user.lower()
    assert CHUNK in user


def test_head_is_truncated_to_head_chars():
    # Документ целиком в промпт не уходит: у GigaChat нет кэша промптов, и
    # платить полным текстом за каждый чанк документа слишком дорого.
    model = FakeModel()
    Contextualizer(model, PROFILE, head_chars=50).context_for(
        DOC, "Описание объекта закупки", CHUNK
    )

    _, user = model.prompts[0]
    assert DOC.parts[0].text[:50] in user
    assert DOC.parts[0].text[:200] not in user


def test_unknown_part_gives_empty_head_not_crash():
    model = FakeModel()
    assert Contextualizer(model, PROFILE).context_for(DOC, "Другого такого нет", CHUNK) == "Раздел про порядок оплаты."


def test_failed_call_is_counted_and_returns_empty_context():
    # Лимит бесплатного тарифа кончается посреди корпуса. Чанк уходит в индекс
    # без контекста, прогон продолжается, счётчик показывает масштаб потерь.
    model = FakeModel(fail=True)
    contextualizer = Contextualizer(model, PROFILE)

    assert contextualizer.context_for(DOC, "Описание объекта закупки", CHUNK) == ""
    assert contextualizer.failures == 1
    assert contextualizer.calls == 0


def test_cache_hit_does_not_call_model_again(tmp_path):
    model = FakeModel()
    contextualizer = Contextualizer(model, PROFILE, cache_path=tmp_path / "contexts.json")

    first = contextualizer.context_for(DOC, "Описание объекта закупки", CHUNK)
    second = contextualizer.context_for(DOC, "Описание объекта закупки", CHUNK)

    assert first == second
    assert len(model.prompts) == 1


def test_cache_survives_new_process(tmp_path):
    path = tmp_path / "contexts.json"
    Contextualizer(FakeModel(), PROFILE, cache_path=path).context_for(DOC, "Описание объекта закупки", CHUNK)

    second_run = Contextualizer(FakeModel(answer="другой ответ"), PROFILE, cache_path=path)
    assert second_run.context_for(DOC, "Описание объекта закупки", CHUNK) == "Раздел про порядок оплаты."
    assert second_run.calls == 0


def test_changed_chunk_text_misses_cache(tmp_path):
    path = tmp_path / "contexts.json"
    contextualizer = Contextualizer(FakeModel(), PROFILE, cache_path=path)

    contextualizer.context_for(DOC, "Описание объекта закупки", CHUNK)
    contextualizer.context_for(DOC, "Описание объекта закупки", CHUNK + " Дополнено.")

    assert contextualizer.calls == 2


def test_cache_key_depends_on_prompt_version():
    # Иначе правка промпта молча подставляла бы старые контексты к новому.
    # Версию промпта задаёт профиль, поэтому смена профиля тоже промахивается
    # мимо чужого кэша, а не приклеивает к тендеру описание из другого домена.
    assert cache_key("1", "doc", "part", "text") != cache_key("2", "doc", "part", "text")


def test_broken_cache_file_does_not_break_ingest(tmp_path):
    path = tmp_path / "contexts.json"
    path.write_text("{это не json", encoding="utf-8")

    contextualizer = Contextualizer(FakeModel(), PROFILE, cache_path=path)
    assert contextualizer.context_for(DOC, "Описание объекта закупки", CHUNK) == "Раздел про порядок оплаты."


def test_cache_writes_readable_utf8(tmp_path):
    path = tmp_path / "contexts.json"
    ContextCache(path).set("ключ", "русский текст")

    assert json.loads(path.read_text(encoding="utf-8")) == {"ключ": "русский текст"}
    # ensure_ascii=False: кэш должен читаться глазами, а не ру.
    assert "русский текст" in path.read_text(encoding="utf-8")


def test_contexts_for_keeps_order_aligned_with_chunks():
    class Counter:
        def __init__(self):
            self.n = 0

        def complete(self, system, user):
            self.n += 1
            return f"контекст {self.n}"

    contexts = Contextualizer(Counter(), PROFILE).contexts_for(
        DOC, "Описание объекта закупки", ["первый", "второй", "третий"]
    )
    assert contexts == ["контекст 1", "контекст 2", "контекст 3"]


class QuotaModel:
    """Тариф кончился: первый же вызов отдаёт QuotaExhausted."""

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        raise QuotaExhausted("402 Payment Required")


def test_exhausted_quota_stops_calling_model():
    # Разница с обычным сбоем: следующий вызов обречён так же, как этот.
    # Пятьсот обречённых запросов подряд стоят минут и не дают ни одного контекста.
    model = QuotaModel()
    contextualizer = Contextualizer(model, PROFILE)

    contexts = contextualizer.contexts_for(
        DOC, "Описание объекта закупки", ["первый", "второй", "третий"]
    )

    assert contexts == ["", "", ""]
    assert model.calls == 1
    assert contextualizer.exhausted is True
    assert contextualizer.skipped == 3
    # Кончившийся тариф - не сбой генерации, и в счётчик сбоев он не идёт:
    # иначе отчёт прогона перестал бы различать «сеть моргнула» и «денег нет».
    assert contextualizer.failures == 0


def test_cache_still_works_after_quota_ran_out(tmp_path):
    # Оплаченное до обрыва должно доехать до индекса: иначе прогон, упавший на
    # лимите, обесценивал бы всё, за что уже заплатили.
    path = tmp_path / "contexts.json"
    Contextualizer(FakeModel(), PROFILE, cache_path=path).context_for(DOC, "Описание объекта закупки", CHUNK)

    after = Contextualizer(QuotaModel(), PROFILE, cache_path=path)
    after.exhausted = True

    assert after.context_for(DOC, "Описание объекта закупки", CHUNK) == "Раздел про порядок оплаты."
    assert after.skipped == 0


def test_contextual_chunks_counts_only_enriched(conn) -> None:
    """Счётчик нужен ровно одной проверке: не выбрасывает ли прогон контексты.

    Индексация с выключенным Contextual Retrieval поверх индекса, собранного с
    ним, тихо пересчитывает эмбеддинги без описаний чанков и роняет поиск.
    Заметить это по выдаче нельзя, поэтому ingest сверяется с этим числом.
    """
    from grounded_rag.config import settings
    from grounded_rag.store import postgres as store

    store.upsert_document(
        conn,
        Document(doc_id="0001", title="Тендер", source_path="/tmp/0001.txt", meta={}),
    )
    vec = [0.0] * settings.embedding_dim
    store.insert_chunk(conn, "0001", "часть", 0, "текст без контекста", vec)
    assert store.contextual_chunks(conn) == 0

    store.insert_chunk(conn, "0001", "часть", 1, "текст", vec, context="Раздел про гардероб")
    assert store.contextual_chunks(conn) == 1


def test_cost_is_counted_before_the_first_call(tmp_path) -> None:
    """Расход считается до прогона, а не по факту.

    Внутри цикла предупреждать поздно: к моменту, когда счёт заметят, деньги
    потрачены. Проверено на себе, замером на тысяче сгенерированных
    документов: полторы тысячи вызовов ушло прежде, чем это стало заметно.
    """
    from grounded_rag.contextual.contextualizer import unpaid_chunks

    cache = ContextCache(tmp_path / "contexts.json")
    planned = {"0001": [("часть", "первый кусок"), ("часть", "второй кусок")]}

    assert unpaid_chunks(cache, planned, PROFILE) == 2

    cache.set(cache_key(PROFILE.prompt_version, "0001", "часть", "первый кусок"), "контекст")
    assert unpaid_chunks(cache, planned, PROFILE) == 1


def test_without_a_cache_everything_is_unpaid() -> None:
    from grounded_rag.contextual.contextualizer import unpaid_chunks

    planned = {"0001": [("часть", "кусок")], "0002": [("часть", "другой")]}

    assert unpaid_chunks(None, planned, PROFILE) == 2
