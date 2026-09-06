"""Служба поиска: что происходит, когда база уезжает у неё из-под ног.

Сама служба это тонкая обёртка над поиском, и проверять в ней особенно нечего,
кроме одного: соединение с Postgres она открывает один раз и держит часами, а
база за это время может перезапуститься. Случай не выдуманный, он случился на
первом же долгом прогоне: Docker подвис, служба осталась жива и до самого
выключения по простою отвечала «the connection is closed» на каждый запрос.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import psycopg
import pytest

SERVE = Path(__file__).resolve().parents[1] / "scripts" / "serve.py"


def load_serve():
    spec = importlib.util.spec_from_file_location("serve_under_test", SERVE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


serve = load_serve()


class DeadConnection:
    """Соединение, которое уже оборвалось и об этом ещё не знает."""

    def __init__(self) -> None:
        self.closed = False

    def execute(self, *args, **kwargs):
        raise psycopg.OperationalError("the connection is closed")

    def close(self) -> None:
        self.closed = True


class LiveConnection:
    def __init__(self, answer: int = 7) -> None:
        self.answer = answer
        self.queries: list[str] = []

    def execute(self, sql, params=None):
        self.queries.append(sql)
        return self

    def fetchone(self):
        return (self.answer,)

    def close(self) -> None:
        pass


@pytest.fixture
def engine(monkeypatch):
    """Служба с подставным соединением, без модели и без настоящей базы."""
    instance = serve.Engine.__new__(serve.Engine)
    instance.conn = DeadConnection()
    return instance


def test_broken_connection_is_reopened_once(engine, monkeypatch):
    fresh = LiveConnection(answer=550)
    monkeypatch.setattr(serve.store, "connect", lambda dsn: fresh)

    assert engine.documents() == 550
    assert engine.conn is fresh, "служба должна остаться с новым соединением"


def test_dead_connection_is_closed_before_reopening(engine, monkeypatch):
    dead = engine.conn
    monkeypatch.setattr(serve.store, "connect", lambda dsn: LiveConnection())

    engine.documents()

    assert dead.closed, "старое соединение надо закрыть, иначе оно течёт"


def test_second_failure_is_not_swallowed(engine, monkeypatch):
    """Повторная попытка ровно одна: если база правда лежит, надо сказать это.

    Молчаливые повторы превратили бы отказ в зависание, а спрашивающая
    программа разбирает причину по тексту ошибки и объясняет её человеку.
    """
    monkeypatch.setattr(serve.store, "connect", lambda dsn: DeadConnection())

    with pytest.raises(psycopg.OperationalError):
        engine.documents()


def test_chunks_of_survives_a_restart_too(engine, monkeypatch):
    """Тот же обрыв на проверке «есть ли документ в индексе».

    Она зовётся отдельным запросом сразу после пустой выдачи, и если её не
    защитить, программа получит отказ ровно там, где решает, показать ли
    человеку «ответа нет» или «нажмите проиндексировать».
    """
    monkeypatch.setattr(serve.store, "connect", lambda dsn: LiveConnection(answer=3))

    assert engine.chunks_of("0152100007026000006") == 3
