"""Один клиент GigaChat на весь процесс.

Раньше вызов модели открывал клиента на каждый запрос: для одного ответа это
незаметно, но контекстуализация корпуса делает вызов на каждый чанк, и 550
авторизационных рукопожатий подряд превращаются в основную стоимость прогона.

Наружу торчит узкий протокол `ChatModel`: контекстуализатору не нужен GigaChat,
ему нужна функция «система плюс запрос -> текст». Поэтому в тестах вместо
модели подставляется заглушка, и ни один тест не ходит в сеть.

Здесь же живёт разделение двух причин отказа, которые снаружи выглядят
одинаково. Слишком частые запросы и упавшая сеть лечатся повтором с паузой.
Кончившийся тариф повтором не лечится, и пятьсот обречённых вызовов подряд
вместо одного отказа - это только потерянное время.
"""

from __future__ import annotations

import time
from typing import Callable, Protocol, TypeVar

T = TypeVar("T")


class QuotaExhausted(RuntimeError):
    """Тариф исчерпан. Повторять бессмысленно, ждать надо пополнения."""


# По каким признакам ответ считается временным. Коды приходят внутри текста
# исключения SDK, отдельного поля со статусом у него нет.
TRANSIENT_MARKERS = ("429", "500", "502", "503", "504", "timeout", "timed out", "connection")
# 402 Payment Required - штатный ответ GigaChat на исчерпанный пакет токенов.
QUOTA_MARKERS = ("402", "payment required", "insufficient")

RETRIES = 3
BASE_DELAY = 2.0


def _looks_like(exc: Exception, markers: tuple[str, ...]) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in markers)


def call_with_retries(
    call: Callable[[], T],
    retries: int = RETRIES,
    base_delay: float = BASE_DELAY,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Повтор с растущей паузой для временных отказов, отдельный тип для тарифа.

    Пауза удваивается (2, 4, 8 секунд), потому что при 429 сервер просит не
    частить, и повтор через сто миллисекунд просит ровно того же ещё раз.

    sleep параметром, чтобы тест на повторы не спал четырнадцать секунд.
    """
    last: Exception | None = None
    for attempt in range(retries):
        try:
            return call()
        except Exception as exc:
            if _looks_like(exc, QUOTA_MARKERS):
                raise QuotaExhausted(str(exc)) from exc
            if not _looks_like(exc, TRANSIENT_MARKERS):
                raise
            last = exc
            if attempt < retries - 1:
                sleep(base_delay * 2**attempt)
    assert last is not None
    raise last


class ChatModel(Protocol):
    """Минимум от LLM: получить ответ на пару (системный промпт, запрос)."""

    def complete(self, system: str, user: str) -> str: ...


class GigaChatModel:
    def __init__(
        self,
        credentials: str,
        scope: str,
        model: str,
        temperature: float = 0.2,
    ) -> None:
        self.credentials = credentials
        self.scope = scope
        self.model = model
        self.temperature = temperature
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from gigachat import GigaChat

            # verify_ssl_certs=False: цепочка сертификатов Сбера подписана
            # «Минцифры России», корневого сертификата которого нет в системном
            # хранилище Windows. Альтернатива - ставить его руками в систему.
            self._client = GigaChat(
                credentials=self.credentials,
                scope=self.scope,
                model=self.model,
                verify_ssl_certs=False,
            )
        return self._client

    def complete(self, system: str, user: str) -> str:
        from gigachat.models import Chat, Messages, MessagesRole

        request = Chat(
            messages=[
                Messages(role=MessagesRole.SYSTEM, content=system),
                Messages(role=MessagesRole.USER, content=user),
            ],
            temperature=self.temperature,
        )
        return call_with_retries(
            lambda: self.client.chat(request).choices[0].message.content
        )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> GigaChatModel:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
