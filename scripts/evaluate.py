"""Гоняет размеченный набор через retrieval и печатает hit@5, MRR и recall@5.

Использование:
    python scripts/evaluate.py [--vector|--rerank] <путь_к_output/docs> [eval/labeled.json]

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
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

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


def main(docs_dir: Path, labeled_path: Path, mode: str = "hybrid") -> None:
    questions = load_questions(labeled_path)

    # Разбирает корпус тот же профиль, что его индексировал: разметка привязана
    # к идентификаторам документов, а их выдаёт именно он.
    problems = check_against_corpus(questions, docs_dir, make_domain(settings))
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

    hits, ranks, recalls = [], [], []
    for question in questions:
        vector = embedder.embed_query(question.query)
        # Глубина больше K: MRR должен различать «нашлось седьмым» и «не нашлось».
        if mode == "vector":
            found = store.search(conn, vector, k=DEPTH)
        else:
            found = retrieve(conn, vector, question.query, k=DEPTH)
        texts = [hit.text for hit in found]

        flags = relevance(texts, question.gold)
        hit = hit_at_k(flags, K)
        rank = reciprocal_rank(flags)
        recall = recall_at_k(texts, question.gold, K)

        hits.append(hit)
        ranks.append(rank)
        recalls.append(recall)

        position = flags.index(True) + 1 if any(flags) else 0
        where = f"позиция {position}" if position else f"не найдено в топ-{DEPTH}"
        mark = "+" if hit else "-"
        print(f"{mark} {question.id:<22} {where:<22} recall@{K} {recall:.2f}   {question.query}")

    total = len(questions)
    print(
        f"\nhit@{K} {sum(hits)}/{total} = {sum(hits) / total:.2f}   "
        f"MRR@{DEPTH} {sum(ranks) / total:.3f}   "
        f"recall@{K} {sum(recalls) / total:.2f}"
    )
    conn.close()


if __name__ == "__main__":
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print("Использование: python scripts/evaluate.py [--vector|--rerank] <docs> [eval/labeled.json]")
        sys.exit(1)
    selected = "vector" if "--vector" in flags else "rerank" if "--rerank" in flags else "hybrid"
    labeled = Path(args[1]) if len(args) > 1 else Path("eval/labeled.json")
    main(Path(args[0]), labeled, selected)
