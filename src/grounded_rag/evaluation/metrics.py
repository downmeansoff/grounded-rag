"""Метрики поиска и разметка, по которой они считаются.

Считать «сколько чанков нужного документа попало в топ-5» можно только там, где
ответ заведомо лежит в одном документе. На запросе «кто заказчик услуг
гардероба» правильных документов все четырнадцать, и это число не значит ничего.

Поэтому разметка привязана не к документу, а к дословному фрагменту текста:
чанк считается релевантным, если содержит фрагмент, который на вопрос и
отвечает. Такую разметку можно проверить - фрагмент либо есть в исходном
документе, либо его там нет, и тогда это ошибка разметки, а не поиска.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


def normalize(text: str) -> str:
    """Схлопывает пробелы и переводы строк.

    Таблицы в документах приезжают с неровными отступами, и один и тот же
    фрагмент в разметке и в чанке отличается только пробелами. Регистр не
    трогаем: он в этих текстах несёт смысл (НМЦК, ОКПД2).
    """
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class Gold:
    """Один эталонный фрагмент: где лежит и как выглядит дословно."""

    doc_id: str
    phrase: str


@dataclass(frozen=True)
class Question:
    id: str
    query: str
    gold: tuple[Gold, ...]


def is_relevant(chunk_text: str, gold: Gold) -> bool:
    return normalize(gold.phrase) in normalize(chunk_text)


def relevance(chunk_texts: list[str], gold: tuple[Gold, ...]) -> list[bool]:
    """Для каждой позиции выдачи: нашёлся ли в ней хоть один эталон."""
    return [any(is_relevant(text, g) for g in gold) for text in chunk_texts]


def hit_at_k(flags: list[bool], k: int) -> bool:
    """Нашёлся ли ответ вообще. Ровно то, что чувствует пользователь."""
    return any(flags[:k])


def reciprocal_rank(flags: list[bool]) -> float:
    """1 / позиция первого верного чанка, 0 если его нет во всей выдаче.

    Отличается от hit тем, что различает первое место и пятое: до первого
    пользователь дочитает всегда, до пятого не всегда.
    """
    for i, flag in enumerate(flags, start=1):
        if flag:
            return 1.0 / i
    return 0.0


def recall_at_k(chunk_texts: list[str], gold: tuple[Gold, ...], k: int) -> float:
    """Доля эталонов, найденных в топ-k.

    Считается по эталонам, а не по позициям: если на вопрос отвечают четыре
    документа, а поиск принёс два, это 0.5, даже когда все пять мест заняты.
    """
    if not gold:
        return 0.0
    found = sum(
        1 for g in gold if any(is_relevant(text, g) for text in chunk_texts[:k])
    )
    return found / len(gold)
