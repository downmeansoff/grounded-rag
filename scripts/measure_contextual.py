"""Меряет, что даёт Contextual Retrieval: одна и та же выдача с контекстом и без.

Замер честный только если обе стороны считаны на одном корпусе и одном
эмбеддере, поэтому скрипт не сравнивает записанные когда-то числа, а
переиндексирует тендер дважды подряд:

    без контекста -> запросы -> с контекстом -> те же запросы -> восстановить

Вторая индексация не платит за LLM: контексты уже лежат в кэше, и повторный
прогон достаёт их оттуда. Первый раз за них платит обычный ingest.

Использование:
    python scripts/measure_contextual.py <путь_к_output/docs> <номер закупки> "запрос" ["запрос" ...]

После прогона тендер остаётся проиндексированным с контекстом.
"""

from __future__ import annotations

import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent))

from grounded_rag.config import settings
from grounded_rag.embed.local import LocalEmbedder
from grounded_rag.store import postgres as store

import ingest_tenders

K = 5


def run_queries(embedder: LocalEmbedder, queries: list[str], reg_number: str) -> dict[str, list[str]]:
    conn = store.connect(settings.dsn)
    results = {}
    for query in queries:
        vector = embedder.embed_query(query)
        hits = store.search_hybrid(conn, vector, query, k=K)
        results[query] = [
            f"{'*' if h.reg_number == reg_number else ' '} {h.reg_number} {h.attachment_name}#{h.chunk_index}"
            for h in hits
        ]
    conn.close()
    return results


def main(docs_dir: Path, reg_number: str, queries: list[str]) -> None:
    embedder = LocalEmbedder(settings.embedding_model, settings.embedding_dim)

    settings.use_contextual = False
    ingest_tenders.main(docs_dir, [reg_number])
    before = run_queries(embedder, queries, reg_number)

    settings.use_contextual = True
    ingest_tenders.main(docs_dir, [reg_number])
    after = run_queries(embedder, queries, reg_number)

    for query in queries:
        print(f"\n=== {query}")
        # Звёздочка слева - чанк искомого тендера. Считать попадания глазами
        # проще, чем доверять одной метрике: видно и порядок, и что вытеснено.
        for i in range(K):
            left = before[query][i] if i < len(before[query]) else ""
            right = after[query][i] if i < len(after[query]) else ""
            mark = "  " if left == right else "->"
            print(f"{i + 1} {mark} без: {left:<58} с контекстом: {right}")

        hit_before = sum(1 for line in before[query] if line.startswith("*"))
        hit_after = sum(1 for line in after[query] if line.startswith("*"))
        print(f"  чанков нужного тендера в топ-{K}: было {hit_before}, стало {hit_after}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print('Использование: python scripts/measure_contextual.py <docs> <номер> "запрос" [...]')
        sys.exit(1)
    main(Path(sys.argv[1]), sys.argv[2], sys.argv[3:])
