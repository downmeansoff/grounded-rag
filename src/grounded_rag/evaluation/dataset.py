"""Чтение размеченного набора и проверка того, что разметка не врёт.

Ноль в метрике значит одно из двух: поиск не нашёл ответ или ответа в корпусе
нет вовсе. Это разные новости, и путать их нельзя, поэтому набор сверяется с
документами до всякого замера.
"""

from __future__ import annotations

import json
from pathlib import Path

from grounded_rag.domain.base import DomainProfile
from grounded_rag.evaluation.metrics import Gold, Question, normalize


def load_questions(path: Path) -> list[Question]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        Question(
            id=q["id"],
            query=q["query"],
            gold=tuple(Gold(g["doc_id"], g["phrase"]) for g in q["gold"]),
        )
        for q in data["questions"]
    ]


def check_against_corpus(
    questions: list[Question], docs_dir: Path, profile: DomainProfile
) -> list[str]:
    """Возвращает список претензий к разметке. Пустой список это порядок.

    Проверяется два условия. Фрагмент должен находиться в документе, за которым
    он записан: иначе идентификатор в разметке проставлен наугад. И фрагмент не
    должен встречаться в других документах: иначе вопрос на самом деле не имеет
    одного правильного ответа, а размечен так, будто имеет.

    Сверка идёт с doc.raw, то есть с файлом целиком, а не с разобранными
    частями. Иначе проверка молчала бы про фрагмент, который в документе есть,
    но который профиль не забрал в parts: это дефект разбора, а разметка тут ни
    при чём, и списывать его на неё нельзя.
    """
    texts = {doc.doc_id: normalize(doc.raw) for doc in profile.load(docs_dir)}

    problems = []
    for question in questions:
        for gold in question.gold:
            phrase = normalize(gold.phrase)
            owners = [doc_id for doc_id, text in texts.items() if phrase in text]
            if gold.doc_id not in owners:
                problems.append(
                    f"{question.id}: фрагмента «{gold.phrase[:50]}...» нет в {gold.doc_id}"
                )
            extra = [doc_id for doc_id in owners if doc_id != gold.doc_id]
            unlabeled = [
                doc_id for doc_id in extra if doc_id not in {g.doc_id for g in question.gold}
            ]
            if unlabeled:
                problems.append(
                    f"{question.id}: фрагмент из {gold.doc_id} встречается ещё в {', '.join(unlabeled)}"
                )
    return problems


def check_against_index(
    questions: list[Question], chunks: list[tuple[str, int, str]]
) -> list[str]:
    """Проверяет, что каждый эталон целиком лежит внутри одного чанка.

    Проверки по документу мало. Чанкер режет текст по абзацам, и фраза может
    попасть на границу: в документе она есть, а в индексе её нет ни в одном
    чанке, и найти её нельзя никаким поиском. Метрика в этом случае показывает
    ноль, который выглядит как провал поиска, хотя это промах разметки мимо
    границы чанка. Разница видна только здесь.
    """
    normalized = [(doc_id, index, normalize(text)) for doc_id, index, text in chunks]

    problems = []
    for question in questions:
        for gold in question.gold:
            phrase = normalize(gold.phrase)
            if not any(
                phrase in text for doc_id, _, text in normalized if doc_id == gold.doc_id
            ):
                problems.append(
                    f"{question.id}: фрагмент «{gold.phrase[:50]}...» есть в документе "
                    f"{gold.doc_id}, но ни в одном его чанке целиком не лежит"
                )
    return problems
