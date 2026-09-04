"""Чтение размеченного набора и проверка того, что разметка не врёт.

Ноль в метрике значит одно из двух: поиск не нашёл ответ или ответа в корпусе
нет вовсе. Это разные новости, и путать их нельзя, поэтому набор сверяется с
документами до всякого замера.
"""

from __future__ import annotations

import json
from pathlib import Path

from grounded_rag.evaluation.metrics import Gold, Question, normalize


def load_questions(path: Path) -> list[Question]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        Question(
            id=q["id"],
            query=q["query"],
            gold=tuple(Gold(g["reg_number"], g["phrase"]) for g in q["gold"]),
        )
        for q in data["questions"]
    ]


def check_against_corpus(questions: list[Question], docs_dir: Path) -> list[str]:
    """Возвращает список претензий к разметке. Пустой список это порядок.

    Проверяется два условия. Фрагмент должен находиться в документе, за которым
    он записан: иначе номер закупки в разметке проставлен наугад. И фрагмент не
    должен встречаться в других документах: иначе вопрос на самом деле не имеет
    одного правильного ответа, а размечен так, будто имеет.
    """
    texts = {
        path.stem: normalize(path.read_text(encoding="utf-8"))
        for path in sorted(docs_dir.glob("*.txt"))
    }

    problems = []
    for question in questions:
        for gold in question.gold:
            phrase = normalize(gold.phrase)
            owners = [reg for reg, text in texts.items() if phrase in text]
            if gold.reg_number not in owners:
                problems.append(
                    f"{question.id}: фрагмента «{gold.phrase[:50]}...» нет в {gold.reg_number}"
                )
            extra = [reg for reg in owners if reg != gold.reg_number]
            unlabeled = [reg for reg in extra if reg not in {g.reg_number for g in question.gold}]
            if unlabeled:
                problems.append(
                    f"{question.id}: фрагмент из {gold.reg_number} встречается ещё в {', '.join(unlabeled)}"
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
    normalized = [(reg, index, normalize(text)) for reg, index, text in chunks]

    problems = []
    for question in questions:
        for gold in question.gold:
            phrase = normalize(gold.phrase)
            if not any(
                phrase in text for reg, _, text in normalized if reg == gold.reg_number
            ):
                problems.append(
                    f"{question.id}: фрагмент «{gold.phrase[:50]}...» есть в документе "
                    f"{gold.reg_number}, но ни в одном его чанке целиком не лежит"
                )
    return problems
