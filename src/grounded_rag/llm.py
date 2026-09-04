"""Один клиент GigaChat на весь процесс.

Раньше вызов модели открывал клиента на каждый запрос: для одного ответа это
незаметно, но контекстуализация корпуса делает вызов на каждый чанк, и 550
авторизационных рукопожатий подряд превращаются в основную стоимость прогона.

Наружу торчит узкий протокол `ChatModel`: контекстуализатору не нужен GigaChat,
ему нужна функция «система плюс запрос -> текст». Поэтому в тестах вместо
модели подставляется заглушка, и ни один тест не ходит в сеть.
"""

from __future__ import annotations

from typing import Protocol


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
        return self.client.chat(request).choices[0].message.content

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> GigaChatModel:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
