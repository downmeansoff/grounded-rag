"""Повторы вокруг вызова LLM: что повторять, что нет, и с какой паузой.

Сеть здесь не нужна: проверяется не GigaChat, а решение, которое принимается
по тексту исключения. 550 вызовов подряд делают это решение дорогим, поэтому
оно должно быть проверено отдельно от всего остального.

sleep подставляется, иначе тест на три попытки спал бы шесть секунд.
"""

from __future__ import annotations

import pytest

from grounded_rag.llm import QuotaExhausted, call_with_retries


class Flaky:
    """Падает первые `fails` раз, потом отдаёт ответ."""

    def __init__(self, fails: int, error: str = "429 Too Many Requests") -> None:
        self.fails = fails
        self.error = error
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        if self.calls <= self.fails:
            raise RuntimeError(self.error)
        return "ответ"


def test_successful_call_does_not_retry():
    call = Flaky(fails=0)
    assert call_with_retries(call, sleep=lambda _: None) == "ответ"
    assert call.calls == 1


def test_transient_failure_is_retried():
    call = Flaky(fails=2)
    assert call_with_retries(call, sleep=lambda _: None) == "ответ"
    assert call.calls == 3


def test_delay_doubles_between_attempts():
    # Пауза растёт, потому что при 429 сервер просит не частить, и повтор через
    # сто миллисекунд просит ровно того же ещё раз. После последней попытки
    # паузы нет: спать перед тем, как сдаться, незачем.
    delays: list[float] = []
    with pytest.raises(RuntimeError):
        call_with_retries(Flaky(fails=99), sleep=delays.append)
    assert delays == [2.0, 4.0]


def test_transient_failure_gives_up_after_retries():
    call = Flaky(fails=99)
    with pytest.raises(RuntimeError, match="429"):
        call_with_retries(call, sleep=lambda _: None)
    assert call.calls == 3


@pytest.mark.parametrize("error", ["402 Payment Required", "insufficient funds"])
def test_quota_is_not_retried(error):
    # Кончившийся тариф повтором не лечится: три попытки вместо одной только
    # растягивают отказ. Отдельный тип нужен, чтобы прогон остановился сам.
    call = Flaky(fails=99, error=error)
    with pytest.raises(QuotaExhausted):
        call_with_retries(call, sleep=lambda _: None)
    assert call.calls == 1


def test_unknown_error_is_not_retried_and_keeps_its_type():
    # Опечатка в промпте или сломанный ответ SDK повтором тоже не лечатся, и
    # прятать их за RuntimeError значит терять след настоящей ошибки.
    def call():
        raise ValueError("model not found")

    with pytest.raises(ValueError, match="model not found"):
        call_with_retries(call, sleep=lambda _: None)
