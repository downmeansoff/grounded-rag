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
символов, где обычно и стоит описание объекта закупки) плюс метаданные тендера.
Это дешевле на два порядка и покрывает главное: чей это документ и о чём он.
"""

from __future__ import annotations

from pathlib import Path

from grounded_rag.contextual.cache import ContextCache, cache_key
from grounded_rag.ingest.loader import TenderDoc
from grounded_rag.llm import ChatModel

# Версия промпта входит в ключ кэша: правка формулировки обесценивает старые
# контексты, и подставлять их к новому промпту было бы враньём.
PROMPT_VERSION = "2"

# Версия 1 перечисляла варианты ответа («раздел, предмет закупки, сторона
# договора, этап») и получала в ответ выбранный вариант одним словом: «Этап»,
# «Ответственность сторон». Для поиска это бесполезно, потому что не приносит
# ни одного слова, которого в чанке ещё нет. Отсюда требование законченного
# предложения, явный список того, что в нём должно прозвучать, и пример.
SYSTEM_PROMPT = (
    "Ты пишешь короткие пояснения к фрагментам тендерной документации, чтобы их "
    "лучше находил поиск. Ответ - одно-два законченных предложения, в которых "
    "обязательно названы предмет закупки, заказчик и раздел документа, откуда "
    "взят фрагмент. Не отвечай одним словом и не отвечай заголовком раздела. "
    "Не пересказывай сам фрагмент и не добавляй вступлений.\n"
    "Пример ответа: «Фрагмент из раздела о порядке приёмки в контракте на уборку "
    "помещений для ГБОУ Школа № 5: описывает сроки подписания акта.»"
)


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
        cache_path: Path | None = None,
        head_chars: int = 1200,
    ) -> None:
        self.model = model
        self.head_chars = head_chars
        self.cache = ContextCache(cache_path) if cache_path else None
        # Вызовы, которые не прошли: лимит тарифа, сеть, отказ модели. Чанк при
        # этом всё равно индексируется, просто без контекста, а счётчик даёт
        # прогону честно сказать, какая часть корпуса осталась необогащённой.
        self.failures = 0
        self.calls = 0

    def _prompt(self, doc: TenderDoc, attachment_name: str, text: str) -> str:
        return (
            f"Тендер: {doc.reg_number}\n"
            f"Название: {doc.title}\n"
            f"Заказчик: {doc.customer}\n"
            f"НМЦК: {doc.price}\n"
            f"Документ: {attachment_name}\n\n"
            f"Начало документа:\n{self.head(doc, attachment_name)}\n\n"
            f"Фрагмент:\n{text}\n\n"
            "Напиши, к чему относится этот фрагмент внутри документа."
        )

    def head(self, doc: TenderDoc, attachment_name: str) -> str:
        for att in doc.attachments:
            if att.name == attachment_name:
                return att.text[: self.head_chars]
        return ""

    def context_for(self, doc: TenderDoc, attachment_name: str, text: str) -> str:
        key = cache_key(PROMPT_VERSION, doc.reg_number, attachment_name, text)
        if self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                return cached

        try:
            context = self.model.complete(SYSTEM_PROMPT, self._prompt(doc, attachment_name, text)).strip()
            self.calls += 1
        except Exception:
            # Обрыв на середине корпуса не должен стоить всего прогона: чанк
            # уходит в индекс без контекста, остальные продолжают обогащаться.
            self.failures += 1
            return ""

        if self.cache is not None:
            self.cache.set(key, context)
        return context

    def contexts_for(self, doc: TenderDoc, attachment_name: str, texts: list[str]) -> list[str]:
        """Строго последовательно: бесплатный тариф GigaChat даёт один поток."""
        return [self.context_for(doc, attachment_name, text) for text in texts]
