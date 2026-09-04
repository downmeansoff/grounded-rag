# grounded-rag

Grounded RAG движок с нуля. Неделя 1 MVP: ingestion → chunking → embedding → Postgres/pgvector → similarity search.

Первый тестовый корпус — тендерная документация (проект [tenderhunt](../тендер)).

## Архитектура

```
ingest.loader      парсит output/docs/*.txt (номер/название/заказчик/НМЦК + вложения)
chunk.recursive     рекурсивный чанкер, 1500/200 символов, режет по абзацам → предложениям → символам
embed.local         intfloat/multilingual-e5-base (локально, без ключа), query/passage префиксы E5
store.postgres      psycopg3 + pgvector, HNSW индекс, cosine distance (<=>)
```

Retrieve/rerank/generation поверх similarity search — следующие шаги, не часть недели 1.

## Запуск

```bash
docker compose up -d
py -3.11 -m venv .venv
.venv/Scripts/pip install -e ".[dev]"
cp .env.example .env
```

Порт Postgres в `.env`/`docker-compose.yml` — `POSTGRES_PORT` (по умолчанию 5433, не 5432 — на этой машине 5432 занят нативным Windows-сервисом postgres.exe).

```bash
python scripts/ingest_tenders.py "C:\Users\glebo\тендер\output\docs"
python scripts/query.py "уборка помещений клининг"
```

## Дальше

- GigaEmbeddings вместо локальной модели (нужен ключ GigaChat API)
- Contextual Retrieval (LLM-контекст перед чанком)
- Hybrid search (BM25 + vector), rerank, generation
