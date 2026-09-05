"""Простой поиск по проиндексированному корпусу.

Использование:
    python scripts/query.py "текст запроса"
    python scripts/query.py "график работы" --filter "Заказчик=музей"
"""

from __future__ import annotations

import argparse
import sys

# stderr тоже: в него argparse пишет ошибку разбора фильтра, а она по-русски.
for stream in (sys.stdout, sys.stderr):
    if stream.encoding != "utf-8":
        stream.reconfigure(encoding="utf-8")

from grounded_rag.config import settings
from grounded_rag.embed.factory import make_embedder
from grounded_rag.retrieve import parse_filters, retrieve
from grounded_rag.store import postgres as store

FILTER_HELP = (
    'ограничить поиск документами, у которых метаданные подходят под "Ключ=значение" '
    "(значение ищется подстрокой без учёта регистра, ключ doc_id сравнивается целиком). "
    "Можно повторять, условия складываются через И"
)


def main(query: str, k: int = 5, filters: dict[str, str] | None = None) -> None:
    embedder = make_embedder(settings)
    conn = store.connect(settings.dsn)

    query_vec = embedder.embed_query(query)
    hits = retrieve(conn, query_vec, query, k=k, filters=filters)

    if not hits:
        where = "" if not filters else " под фильтр " + ", ".join(f"{key}={value}" for key, value in filters.items())
        print(f"Ничего не найдено{where}.")
        return

    for i, hit in enumerate(hits, 1):
        score = "" if hit.rerank_score is None else f" rerank={hit.rerank_score:.4f}"
        print(f"\n[{i}] distance={hit.distance:.4f}{score} doc={hit.doc_id} part={hit.part_name}#{hit.chunk_index}")
        print(f"    {hit.title}")
        snippet = hit.text.strip().replace("\n", " ")
        print(f"    {snippet[:300]}")

    conn.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Поиск по корпусу.")
    parser.add_argument("query", help="текст запроса")
    parser.add_argument("-k", type=int, default=5, help="сколько чанков показать (по умолчанию 5)")
    parser.add_argument("--filter", dest="filters", action="append", metavar="КЛЮЧ=ЗНАЧЕНИЕ", help=FILTER_HELP)

    args = parser.parse_args()
    try:
        args.filters = parse_filters(args.filters)
    except ValueError as error:
        parser.error(str(error))
    return args


if __name__ == "__main__":
    args = _parse_args()
    main(args.query, args.k, args.filters)
