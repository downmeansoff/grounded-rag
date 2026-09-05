"""Ответ с цитатами по проиндексированному корпусу (retrieval + генерация).

Использование:
    python scripts/ask.py "какой график работы?"
    python scripts/ask.py "какой график работы?" --filter "Заказчик=музей"
    python scripts/ask.py "какой штраф в тюменской поликлинике?" --auto-filter
"""

from __future__ import annotations

import argparse
import sys

# stderr тоже: в него argparse пишет ошибку разбора фильтра, а она по-русски.
for stream in (sys.stdout, sys.stderr):
    if stream.encoding != "utf-8":
        stream.reconfigure(encoding="utf-8")

from grounded_rag.autofilter import auto_filter, explain
from grounded_rag.config import settings
from grounded_rag.domain.factory import make_domain
from grounded_rag.embed.factory import make_embedder
from grounded_rag.generate.gigachat_llm import answer
from grounded_rag.retrieve import parse_filters, retrieve
from grounded_rag.store import postgres as store

FILTER_HELP = (
    'искать ответ только в документах, у которых метаданные подходят под "Ключ=значение" '
    "(значение ищется подстрокой без учёта регистра, ключ doc_id сравнивается целиком). "
    "Можно повторять, условия складываются через И"
)
AUTO_FILTER_HELP = (
    "достать имя заказчика из текста вопроса и подставить его в фильтр по метаданным. "
    "Заданный руками --filter по тому же ключу сильнее"
)


def main(query: str, k: int = 5, filters: dict[str, str] | None = None, auto: bool = False) -> None:
    profile = make_domain(settings)
    embedder = make_embedder(settings)
    conn = store.connect(settings.dsn)

    if auto:
        filters, found = auto_filter(conn, query, profile.filter_key, filters)
        print(explain(profile.name, profile.filter_key, found))

    query_vec = embedder.embed_query(query)
    hits = retrieve(conn, query_vec, query, k=k, filters=filters)
    conn.close()

    if not hits:
        # Пустая выдача под фильтром это не отсутствие ответа в корпусе, а
        # отсутствие документов, подходящих под фильтр. Разница важная: иначе
        # опечатка в имени заказчика читается как "в документах такого нет".
        where = "" if not filters else " под фильтр " + ", ".join(f"{key}={value}" for key, value in filters.items())
        print(f"Ничего не найдено{where}.")
        return

    text = answer(
        query,
        hits,
        profile,
        settings.gigachat_credentials,
        settings.gigachat_scope,
        settings.gigachat_model,
    )
    print(text)

    print("\nИсточники:")
    for i, hit in enumerate(hits, 1):
        citation = profile.citation(hit.doc_id, hit.part_name, hit.chunk_index)
        print(f"  [{i}] {citation} - {hit.title}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ответ с цитатами по корпусу.")
    parser.add_argument("query", help="вопрос")
    parser.add_argument("-k", type=int, default=5, help="сколько чанков отдать модели (по умолчанию 5)")
    parser.add_argument("--filter", dest="filters", action="append", metavar="КЛЮЧ=ЗНАЧЕНИЕ", help=FILTER_HELP)
    parser.add_argument("--auto-filter", dest="auto", action="store_true", help=AUTO_FILTER_HELP)

    args = parser.parse_args()
    try:
        args.filters = parse_filters(args.filters)
    except ValueError as error:
        parser.error(str(error))
    return args


if __name__ == "__main__":
    args = _parse_args()
    main(args.query, args.k, args.filters, args.auto)
