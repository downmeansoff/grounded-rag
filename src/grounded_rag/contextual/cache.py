"""Кэш сгенерированных контекстов на диске.

Контекст стоит одного вызова LLM на чанк. На корпусе в 550 чанков это 550
вызовов, а бесплатный тариф GigaChat даёт один поток и конечный запас токенов,
так что повторный ingest без кэша - это повторная оплата уже сделанной работы.

Ключ считается от того, что реально уходит в промпт: номер закупки, имя
вложения, текст чанка и версия самого промпта. Меняется чанкер или промпт -
ключ меняется, и старый контекст не подставляется молча к новому тексту.

Запись атомарная (временный файл плюс os.replace) и сразу после каждого нового
контекста: прогон обрывается на исчерпании лимита в середине, и терять из-за
этого триста уже оплаченных ответов нельзя.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def cache_key(prompt_version: str, doc_id: str, part_name: str, text: str) -> str:
    payload = "\x00".join([prompt_version, doc_id, part_name, text])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ContextCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: dict[str, str] = {}
        if path.exists():
            try:
                self._data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                # Битый кэш - не повод ронять ingest: он пересоберётся сам.
                self._data = {}

    def __len__(self) -> int:
        return len(self._data)

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def set(self, key: str, value: str) -> None:
        self._data[key] = value
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._data, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)
