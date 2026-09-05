"""Гоняет размеченный набор через retrieval и печатает hit@5, MRR и recall@5.

Использование:
    python scripts/evaluate.py [--vector|--rerank] [--auto-filter] <путь_к_output/docs> [eval/labeled.json]

Сначала разметка сверяется с документами: фрагмент обязан лежать в том документе,
за которым записан, и не встречаться в чужих. Не сошлось - прогон падает, а не
показывает ноль: ноль от плохого поиска и ноль от кривой разметки выглядят
одинаково, а значат разное.

Дальше каждый вопрос идёт через ту же функцию retrieve, что и `ask.py`, поэтому
замер меряет рабочий поиск, а не отдельную его копию для красивых чисел.

Режимы отличаются одной ступенью каждый, чтобы было видно вклад именно её:
`--vector` только эмбеддинги, по умолчанию гибрид с полнотекстом, `--rerank`
гибрид плюс cross-encoder. Индекс при этом один и тот же, переиндексация не
нужна.

Вопросы с фильтром по метаданным идут отдельным блоком и в общие числа не
входят: фильтр сужает корпус до одного заказчика, попасть по такому вопросу
легче, и общая средняя выросла бы от работы, которой поиск не делал. Рядом
печатается позиция того же вопроса без фильтра, потому что смысл фильтра
именно в разнице между этими двумя.

`--auto-filter` включает подстановку заказчика из текста вопроса в фильтр. Тогда
обычные вопросы идут уже с ним, и общие числа прямо сравнимы с прогоном без
флага: набор вопросов, индекс и режим поиска те же, отличается одна ступень. У
вопросов с ручным фильтром рядом печатается, достал ли автофильтр того же
заказчика: там правильный ответ известен заранее.
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from grounded_rag.autofilter import auto_filter, common_lexemes, customer_lexemes, short_name
from grounded_rag.config import settings
from grounded_rag.domain.factory import make_domain
from grounded_rag.embed.factory import make_embedder
from grounded_rag.evaluation.dataset import (
    check_against_corpus,
    check_against_index,
    load_questions,
)
from grounded_rag.evaluation.metrics import (
    hit_at_k,
    recall_at_k,
    reciprocal_rank,
    relevance,
)
from grounded_rag.retrieve import retrieve
from grounded_rag.store import postgres as store

K = 5
DEPTH = 10


def report(problems: list[str], header: str) -> None:
    """Печатает претензии к разметке и обрывает прогон.

    Именно обрывает, а не показывает ноль: ноль от плохого поиска и ноль от
    кривой разметки выглядят одинаково, а значат разное.
    """
    print(header)
    for problem in problems:
        print("  ", problem)
    sys.exit(1)


def search(conn, embedder, question, mode: str, filters: dict[str, str] | None) -> list[str]:
    """Тексты выдачи по вопросу, сверху вниз.

    Глубина больше K: MRR должен различать «нашлось седьмым» и «не нашлось».
    """
    vector = embedder.embed_query(question.query)
    if mode == "vector":
        found = store.search(conn, vector, k=DEPTH, filters=filters)
    else:
        found = retrieve(conn, vector, question.query, k=DEPTH, filters=filters)
    return [hit.text for hit in found]


def where(flags: list[bool]) -> str:
    position = flags.index(True) + 1 if any(flags) else 0
    return f"позиция {position}" if position else f"не найдено в топ-{DEPTH}"


def totals(rows: list[tuple[bool, float, float]], header: str) -> None:
    total = len(rows)
    print(
        f"\n{header}hit@{K} {sum(h for h, _, _ in rows)}/{total} = "
        f"{sum(h for h, _, _ in rows) / total:.2f}   "
        f"MRR@{DEPTH} {sum(r for _, r, _ in rows) / total:.3f}   "
        f"recall@{K} {sum(c for _, _, c in rows) / total:.2f}"
    )


def main(docs_dir: Path, labeled_path: Path, mode: str = "hybrid", auto: bool = False) -> None:
    questions = load_questions(labeled_path)
    profile = make_domain(settings)

    # Разбирает корпус тот же профиль, что его индексировал: разметка привязана
    # к идентификаторам документов, а их выдаёт именно он.
    problems = check_against_corpus(questions, docs_dir, profile)
    if problems:
        report(problems, "Разметка не сходится с корпусом:")

    embedder = make_embedder(settings)
    conn = store.connect(settings.dsn)

    # Второй раз то же самое, но по индексу: фраза может лежать в документе и
    # при этом разваливаться по границе чанков, а искать поиск умеет только
    # чанки целиком.
    problems = check_against_index(questions, store.all_chunk_texts(conn))
    if problems:
        report(problems, "Разметка не сходится с индексом:")

    print(f"Разметка проверена по корпусу и по индексу: вопросов {len(questions)}, "
          f"эталонных фрагментов {sum(len(q.gold) for q in questions)}")

    settings.use_rerank = mode == "rerank"
    print(f"Режим поиска: {mode}")

    # Вопросы с фильтром считаются отдельно и в общие числа не идут. Фильтр
    # сужает корпус до одного заказчика, попасть по такому вопросу заметно
    # легче, и подмешивать его в ту же среднюю значило бы поднять её работой,
    # которую поиск не делал.
    plain = [q for q in questions if not q.filters]
    filtered = [q for q in questions if q.filters]

    # Список заказчиков читается один раз на весь прогон: от вопроса к вопросу
    # он не меняется, а вопросов полсотни.
    key = profile.filter_key if auto else ""
    known = customer_lexemes(conn, key) if key else {}
    common = common_lexemes(conn, known) if key else set()
    if auto:
        print(
            f"Автофильтр включён: заказчик берётся из текста вопроса, ключ {key or 'не задан профилем'}"
            if key
            else f'Автофильтр включён, но профиль "{profile.name}" не хранит заказчика в метаданных'
        )

    rows = []
    for question in plain:
        pairs, found = auto_filter(conn, question.query, key, None, known, common) if key else ({}, None)
        texts = search(conn, embedder, question, mode, pairs or None)
        flags = relevance(texts, question.gold)
        row = (
            hit_at_k(flags, K),
            reciprocal_rank(flags),
            recall_at_k(texts, question.gold, K),
        )
        rows.append(row)
        mark = "+" if row[0] else "-"
        note = "" if not auto else f"   [{short_name(found) if found else 'заказчик не опознан'}]"
        print(
            f"{mark} {question.id:<22} {where(flags):<22} "
            f"recall@{K} {row[2]:.2f}   {question.query}{note}"
        )
    totals(rows, "")

    if not filtered:
        conn.close()
        return

    # Рядом с каждым фильтром печатается та же выдача без него: смысл фильтра
    # в разнице между этими двумя строками, а не в самой по себе позиции.
    print("\nВопросы с фильтром по метаданным (в числа выше не входят):")
    rows = []
    for question in filtered:
        pairs = dict(question.filters)
        without = relevance(search(conn, embedder, question, mode, None), question.gold)
        texts = search(conn, embedder, question, mode, pairs)
        flags = relevance(texts, question.gold)
        row = (
            hit_at_k(flags, K),
            reciprocal_rank(flags),
            recall_at_k(texts, question.gold, K),
        )
        rows.append(row)
        mark = "+" if row[0] else "-"
        shown = ", ".join(f"{name}={value}" for name, value in pairs.items())
        # Ручной фильтр тут и есть эталон для автофильтра: у этих вопросов
        # известно, какого заказчика надо было достать из текста.
        note = ""
        if key:
            _, guess = auto_filter(conn, question.query, key, None, known, common)
            hand = pairs.get(key, "").lower()
            if guess and hand and hand in guess.lower():
                note = "   автофильтр: тот же заказчик"
            elif guess:
                note = f"   автофильтр: другой заказчик ({short_name(guess, 30)})"
            else:
                note = "   автофильтр: не опознал"
        print(
            f"{mark} {question.id:<30} без фильтра: {where(without):<22} "
            f"с фильтром: {where(flags):<14} {shown}{note}"
        )
    totals(rows, "с фильтром: ")
    conn.close()


if __name__ == "__main__":
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print(
            "Использование: python scripts/evaluate.py [--vector|--rerank] [--auto-filter] "
            "<docs> [eval/labeled.json]"
        )
        sys.exit(1)
    selected = "vector" if "--vector" in flags else "rerank" if "--rerank" in flags else "hybrid"
    labeled = Path(args[1]) if len(args) > 1 else Path("eval/labeled.json")
    main(Path(args[0]), labeled, selected, "--auto-filter" in flags)
