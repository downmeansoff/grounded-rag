"""Contextual Retrieval: перед эмбеддингом чанк получает своё место в документе.

Проблема, которую это лечит. Чанкер режет документ по абзацам, и середина
документа приезжает в индекс без единого признака того, откуда она. Фрагмент
«Оплата производится в течение 15 рабочих дней с даты подписания» одинаково
выглядит в контракте на уборку и в контракте на охрану, а фрагмент «Пункт 4.2.
Штраф составляет 5% от цены этапа» вообще не содержит слова, по которому его
стали бы искать. Поиск при этом идёт по эмбеддингу и по токенам самого чанка,
так что взять недостающее ему неоткуда.

Решение: перед индексацией спросить у LLM одно-два предложения о том, к чему
этот фрагмент относится, и приклеить их к тексту. В индекс идёт склейка, в
цитату - оригинальный текст: пользователь должен видеть документ, а не пересказ.

Отличие от варианта Anthropic. В оригинале в промпт кладут документ целиком и
экономят на prompt caching. У GigaChat кэша промптов нет, а бесплатный тариф
конечен, поэтому вместо всего документа уходит его шапка (первые head_chars
символов, где обычно и стоит описание предмета) плюс метаданные документа.
Это дешевле на два порядка и покрывает главное: чей это документ и о чём он.
"""

from __future__ import annotations

from pathlib import Path

from grounded_rag.contextual.cache import ContextCache, cache_key
from grounded_rag.domain.base import Document, DomainProfile
from grounded_rag.llm import ChatModel, QuotaExhausted

# Сами формулировки живут в профиле предметной области: они единственная часть
# контекстуализации, которая зависит от того, что именно индексируется. Оттуда
# же берётся версия промпта, и она входит в ключ кэша - правка формулировки
# обесценивает старые контексты, и подставлять их к новому промпту было бы
# враньём.


def enrich(context: str, text: str) -> str:
    """Что уходит в эмбеддер: контекст перед текстом или текст как есть.

    Склейка живёт одной функцией, потому что её должны одинаково считать и
    ingest, и тесты. Разъедутся - индекс перестанет соответствовать тому, что
    считает индексом остальной код.
    """
    context = context.strip()
    return f"{context}\n\n{text}" if context else text


class Contextualizer:
    def __init__(
        self,
        model: ChatModel,
        profile: DomainProfile,
        cache_path: Path | None = None,
        head_chars: int = 1200,
    ) -> None:
        self.model = model
        self.profile = profile
        self.head_chars = head_chars
        self.cache = ContextCache(cache_path) if cache_path else None
        # Вызовы, которые не прошли: сеть, отказ модели. Чанк при этом всё равно
        # индексируется, просто без контекста, а счётчик даёт прогону честно
        # сказать, какая часть корпуса осталась необогащённой.
        self.failures = 0
        self.calls = 0
        # Кончившийся тариф отличается от сетевого сбоя тем, что следующий вызов
        # обречён так же, как этот. Поэтому после первого такого отказа модель
        # больше не дёргается, а оставшиеся чанки считаются пропущенными: пятьсот
        # обречённых запросов подряд стоят минут и не приносят ни одного контекста.
        self.exhausted = False
        self.skipped = 0

    def head(self, doc: Document, part_name: str) -> str:
        part = doc.part(part_name)
        return part.text[: self.head_chars] if part else ""

    def context_for(self, doc: Document, part_name: str, text: str) -> str:
        key = cache_key(self.profile.prompt_version, doc.doc_id, part_name, text)
        if self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                return cached

        if self.exhausted:
            self.skipped += 1
            return ""

        try:
            prompt = self.profile.context_prompt(doc, part_name, self.head(doc, part_name), text)
            context = self.model.complete(self.profile.context_system, prompt).strip()
            self.calls += 1
        except QuotaExhausted:
            self.exhausted = True
            self.skipped += 1
            return ""
        except Exception:
            # Обрыв на середине корпуса не должен стоить всего прогона: чанк
            # уходит в индекс без контекста, остальные продолжают обогащаться.
            self.failures += 1
            return ""

        if self.cache is not None:
            self.cache.set(key, context)
        return context

    def contexts_for(self, doc: Document, part_name: str, texts: list[str]) -> list[str]:
        """Строго последовательно: бесплатный тариф GigaChat даёт один поток."""
        return [self.context_for(doc, part_name, text) for text in texts]
