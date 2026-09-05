"""Профиль под закупочную документацию: формат выгрузки tenderhunt.

Один файл это одна закупка. Сверху шапка «КЛЮЧ: значение», дальше строка из
знаков равенства, дальше приложения, каждое под заголовком `### имя [ext]`.
Так его пишет соседний проект, и профиль знает ровно этот формат.

Ключи шапки русские и заглавными, потому что такими их выгружает источник.
НОМЕР становится идентификатором документа, ЗАКАЗЧИК и НМЦК уезжают в
метаданные и оттуда в промпт контекстуализации: без заказчика описание чанка
получается верным, но одинаковым для всех четырнадцати закупок сразу.

Промпты здесь дословно те же, что были до появления профилей, и версия та же.
Это не аккуратность ради аккуратности: версия входит в ключ кэша контекстов,
и любая правка формулировки обесценила бы уже оплаченные пятьсот с лишним
вызовов LLM.
"""

from __future__ import annotations

import re
from pathlib import Path

from grounded_rag.domain.base import Document, DomainProfile, Part

_SEPARATOR = re.compile(r"^=+$", re.MULTILINE)
_HEADER_LINE = re.compile(r"^([А-ЯA-Z,() ]+): (.*)$", re.MULTILINE)
_SECTION_HEADER = re.compile(r"^### (.+) \[(\w+)\]$", re.MULTILINE)


def _parse_header(raw: str) -> dict[str, str]:
    return {k.strip(): v.strip() for k, v in _HEADER_LINE.findall(raw)}


def _parse_sections(body: str) -> list[Part]:
    matches = list(_SECTION_HEADER.finditer(body))
    parts = []
    for i, m in enumerate(matches):
        name, ext = m.group(1), m.group(2)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        text = body[start:end].strip()
        if text:
            parts.append(Part(name=name, text=text, ext=ext))
    return parts


def parse_file(path: Path) -> Document | None:
    raw = path.read_text(encoding="utf-8", errors="replace")
    split = _SEPARATOR.split(raw, maxsplit=1)
    header_raw = split[0]
    body = split[1] if len(split) > 1 else ""

    header = _parse_header(header_raw)
    return Document(
        doc_id=header.get("НОМЕР", path.stem),
        title=header.get("НАЗВАНИЕ", ""),
        source_path=str(path),
        meta={
            "Заказчик": header.get("ЗАКАЗЧИК", ""),
            "НМЦК": header.get("НМЦК", ""),
        },
        parts=_parse_sections(body),
        raw=raw,
    )


class TendersProfile(DomainProfile):
    name = "tenders"
    entity = "тендер"
    corpus = "тендерной документации"
    prompt_version = "2"
    filter_key = "Заказчик"

    def load(self, docs_dir: Path) -> list[Document]:
        docs = []
        for path in sorted(docs_dir.glob("*.txt")):
            doc = parse_file(path)
            if doc and doc.parts:
                docs.append(doc)
        return docs

    @property
    def context_system(self) -> str:
        # Версия 1 перечисляла варианты ответа («раздел, предмет закупки,
        # сторона договора, этап») и получала в ответ выбранный вариант одним
        # словом. Для поиска это бесполезно: слов, которых нет в самом чанке,
        # такой ответ не приносит.
        return (
            "Ты пишешь короткие пояснения к фрагментам тендерной документации, чтобы их "
            "лучше находил поиск. Ответ - одно-два законченных предложения, в которых "
            "обязательно названы предмет закупки, заказчик и раздел документа, откуда "
            "взят фрагмент. Не отвечай одним словом и не отвечай заголовком раздела. "
            "Не пересказывай сам фрагмент и не добавляй вступлений.\n"
            "Пример ответа: «Фрагмент из раздела о порядке приёмки в контракте на уборку "
            "помещений для ГБОУ Школа № 5: описывает сроки подписания акта.»"
        )

    def context_prompt(self, doc: Document, part_name: str, head: str, text: str) -> str:
        meta = "".join(f"{key}: {value}\n" for key, value in doc.meta.items())
        return (
            f"Тендер: {doc.doc_id}\n"
            f"Название: {doc.title}\n"
            f"{meta}"
            f"Документ: {part_name}\n\n"
            f"Начало документа:\n{head}\n\n"
            f"Фрагмент:\n{text}\n\n"
            "Напиши, к чему относится этот фрагмент внутри документа."
        )
