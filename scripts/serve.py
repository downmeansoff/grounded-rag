"""Движок как локальная служба: модель грузится один раз, а не на каждый вопрос.

Запуск скрипта на каждый запрос стоит около двадцати шести секунд, из которых
сам поиск занимает десятки миллисекунд. Всё остальное это импорт torch и
загрузка модели эмбеддингов. Для командной строки это терпимо, для окна
программы нет: человек ждёт полминуты и решает, что оно сломано.

Служба держит модель и соединение с базой в памяти и отвечает за те самые
десятки миллисекунд.

    python scripts/serve.py [порт]

Слушает только localhost и не спрашивает пароля: это инструмент на своей
машине, рядом с базой, которая тоже слушает localhost. Выставлять его наружу
нельзя, потому что тогда любой в сети получит доступ к содержимому индекса.

Служба выключается сама после IDLE_TIMEOUT без запросов. Так процесс с моделью
в памяти не остаётся висеть, если программа, которая его подняла, закрылась
или упала.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import psycopg

from grounded_rag.config import settings
from grounded_rag.domain.factory import make_domain
from grounded_rag.embed.factory import make_embedder
from grounded_rag.retrieve import retrieve
from grounded_rag.store import postgres as store

DEFAULT_PORT = 8799
IDLE_TIMEOUT = 30 * 60
MAX_BODY = 64 * 1024

_last_request = time.monotonic()
_lock = threading.Lock()


class Engine:
    """Модель, профиль и соединение с базой, живущие всё время работы службы."""

    def __init__(self) -> None:
        # База первой, модель второй. Порядок не косметический: модель грузится
        # полминуты, а база или отвечает сразу, или не отвечает вовсе. При
        # обратном порядке отказ «нет базы» приходит через сорок секунд вместо
        # пяти, и всё это время программа, которая подняла службу, не может
        # сказать человеку, что достаточно запустить Docker.
        self.profile = make_domain(settings)
        self.conn = store.connect(settings.dsn)
        self.embedder = make_embedder(settings)

    def retrying(self, work):
        """Работа с базой, переживающая обрыв соединения: одна повторная попытка.

        Соединение открывается один раз и живёт часами, а Postgres под ним
        может уехать: контейнер перезапустили, Docker Desktop подвис, машина
        уснула. Так и случилось при первом же долгом прогоне, и служба после
        этого осталась жива, но на каждый запрос отвечала «the connection is
        closed» до самого выключения по простою. Человек в окне видел поломку
        поиска там, где достаточно переподключиться.
        """
        try:
            return work()
        except psycopg.OperationalError:
            try:
                self.conn.close()
            except Exception:  # соединение и так мертво, закрывать нечего
                pass
            self.conn = store.connect(settings.dsn)
            return work()

    def documents(self) -> int:
        with _lock:
            return self.retrying(
                lambda: self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            )

    def chunks_of(self, doc_id: str) -> int:
        """Сколько чанков у документа. Ноль означает «не индексировали».

        Нужно спрашивающей программе, чтобы отличить «в документе нет ответа»
        от «документ ещё не в индексе»: выдача в обоих случаях пустая, а делать
        человеку надо разное.
        """
        with _lock:
            row = self.retrying(
                lambda: self.conn.execute(
                    "SELECT COUNT(*) FROM chunks WHERE doc_id = %s", (doc_id,)
                ).fetchone()
            )
        return row[0] if row else 0

    def search(self, question: str, filters: dict[str, str] | None, k: int) -> list[dict]:
        # Соединение одно на службу, а запросы могут прийти одновременно:
        # psycopg не обещает потокобезопасности на одном соединении.
        with _lock:
            vector = self.embedder.embed_query(question)
            hits = self.retrying(
                lambda: retrieve(self.conn, vector, question, k=k, filters=filters or None)
            )
        return [
            {
                "doc_id": hit.doc_id,
                "title": hit.title,
                "part_name": hit.part_name,
                "chunk_index": hit.chunk_index,
                "citation": self.profile.citation(hit.doc_id, hit.part_name, hit.chunk_index),
                "distance": round(hit.distance, 4),
                "rerank_score": hit.rerank_score,
                "text": hit.text,
            }
            for hit in hits
        ]


class Handler(BaseHTTPRequestHandler):
    engine: Engine

    def log_message(self, *args) -> None:  # noqa: D102 - в консоль пишет только служба
        pass

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - имя задаёт http.server
        global _last_request
        _last_request = time.monotonic()
        path, _, query = self.path.partition("?")
        if path.rstrip("/") == "/document":
            doc_id = parse_qs(query).get("doc_id", [""])[0]
            if not doc_id:
                self._send(400, {"error": "не задан doc_id"})
                return
            chunks = self.engine.chunks_of(doc_id)
            self._send(200, {"doc_id": doc_id, "indexed": chunks > 0, "chunks": chunks})
            return

        if path.rstrip("/") != "/health":
            self._send(404, {"error": "нет такого маршрута"})
            return
        self._send(200, {"ok": True, "documents": self.engine.documents()})

    def do_POST(self) -> None:  # noqa: N802 - имя задаёт http.server
        global _last_request
        _last_request = time.monotonic()
        if self.path.rstrip("/") != "/search":
            self._send(404, {"error": "нет такого маршрута"})
            return

        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            self._send(413, {"error": "слишком длинный запрос"})
            return
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"error": "тело запроса не JSON"})
            return

        question = str(payload.get("question") or "").strip()
        if not question:
            self._send(400, {"error": "пустой вопрос"})
            return

        try:
            hits = self.engine.search(
                question,
                dict(payload.get("filters") or {}),
                int(payload.get("k") or 5),
            )
        except Exception as error:  # служба не должна падать от одного запроса
            self._send(500, {"error": f"{type(error).__name__}: {error}"})
            return
        self._send(200, {"question": question, "hits": hits})


def _watchdog(server: ThreadingHTTPServer) -> None:
    while True:
        time.sleep(30)
        if time.monotonic() - _last_request > IDLE_TIMEOUT:
            print("Простой дольше положенного, служба выключается.")
            threading.Thread(target=server.shutdown, daemon=True).start()
            return


def main(port: int = DEFAULT_PORT) -> None:
    print("Загружаю модель и подключаюсь к базе...")
    Handler.engine = Engine()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Служба поиска слушает http://127.0.0.1:{port}, документов в индексе: "
          f"{Handler.engine.documents()}")
    threading.Thread(target=_watchdog, args=(server,), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT)
