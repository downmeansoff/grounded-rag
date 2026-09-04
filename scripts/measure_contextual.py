"""Меряет, что даёт Contextual Retrieval: одна и та же выдача с контекстом и без.

Замер честный только если обе стороны считаны на одном корпусе и одном
эмбеддере, поэтому скрипт не сравнивает записанные когда-то числа, а
переиндексирует документ дважды подряд:

    без контекста -> запросы -> с контекстом -> те же запросы -> восстановить

Вторая индексация не платит за LLM: контексты уже лежат в кэше, и повторный
прогон достаёт их оттуда. Первый раз за них платит обычный ingest.

Использование:
    python scripts/measure_contextual.py [--all] <путь_к_документам> <идентификатор> "запрос" ["запрос" ...]

Идентификатор документа задаёт, чьи чанки считать попаданием: они помечаются
звёздочкой. Без --all переиндексируется только этот документ, и он соревнуется
без контекста с уже обогащёнными соседями. С --all контекст снимается и
возвращается всему корпусу сразу, и это честнее: сравниваются два состояния
всего индекса.

После прогона корпус остаётся проиндексированным с контекстом.
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent))

from grounded_rag.config import settings
from grounded_rag.embed.base import Embedder
from grounded_rag.embed.factory import make_embedder
from grounded_rag.store import postgres as store

import ingest

K = 5


def run_queries(embedder: Embedder, queries: list[str], doc_id: str) -> dict[str, list[str]]:
    conn = store.connect(settings.dsn)
    results = {}
    for query in queries:
        vector = embedder.embed_query(query)
        hits = store.search_hybrid(conn, vector, query, k=K)
        results[query] = [
            f"{'*' if h.doc_id == doc_id else ' '} {h.doc_id} {h.part_name}#{h.chunk_index}"
            for h in hits
        ]
    conn.close()
    return results


def main(docs_dir: Path, doc_id: str, queries: list[str], whole_corpus: bool = False) -> None:
    embedder = make_embedder(settings)
    only = None if whole_corpus else [doc_id]

    settings.use_contextual = False
    ingest.main(docs_dir, only)
    before = run_queries(embedder, queries, doc_id)

    settings.use_contextual = True
    ingest.main(docs_dir, only)
    after = run_queries(embedder, queries, doc_id)

    for query in queries:
        print(f"\n=== {query}")
        # Звёздочка слева - чанк искомого документа. Считать попадания глазами
        # проще, чем доверять одной метрике: видно и порядок, и что вытеснено.
        for i in range(K):
            left = before[query][i] if i < len(before[query]) else ""
            right = after[query][i] if i < len(after[query]) else ""
            mark = "  " if left == right else "->"
            print(f"{i + 1} {mark} без: {left:<58} с контекстом: {right}")

        hit_before = sum(1 for line in before[query] if line.startswith("*"))
        hit_after = sum(1 for line in after[query] if line.startswith("*"))
        print(f"  чанков нужного документа в топ-{K}: было {hit_before}, стало {hit_after}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--all"]
    if len(args) < 3:
        print('Использование: python scripts/measure_contextual.py [--all] <docs> <идентификатор> "запрос" [...]')
        sys.exit(1)
    main(Path(args[0]), args[1], args[2:], whole_corpus="--all" in sys.argv)
