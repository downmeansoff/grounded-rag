"""Читает корпус тендерной документации, собранный tenderhunt в output/docs/*.txt.

Формат одного файла (см. tenderhunt/parsing/docs.py и pipeline.py):
    НОМЕР: <reg_number>
    НАЗВАНИЕ: <title>
    ЗАКАЗЧИК: <customer>
    НМЦК: <price>
    ...
    ======================================================================

    ### <имя вложения> [<расширение>]
    <извлечённый текст вложения>

    ### <следующее вложение> [<ext>]
    <текст>
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_SEPARATOR = re.compile(r"^=+$", re.MULTILINE)
_HEADER_LINE = re.compile(r"^([А-ЯA-Z,() ]+): (.*)$", re.MULTILINE)
_SECTION_HEADER = re.compile(r"^### (.+) \[(\w+)\]$", re.MULTILINE)


@dataclass
class Attachment:
    reg_number: str
    name: str
    ext: str
    text: str


@dataclass
class TenderDoc:
    reg_number: str
    title: str = ""
    customer: str = ""
    price: str = ""
    source_path: str = ""
    attachments: list[Attachment] = field(default_factory=list)


def _parse_header(raw: str) -> dict[str, str]:
    return {k.strip(): v.strip() for k, v in _HEADER_LINE.findall(raw)}


def _parse_sections(body: str, reg_number: str) -> list[Attachment]:
    matches = list(_SECTION_HEADER.finditer(body))
    attachments = []
    for i, m in enumerate(matches):
        name, ext = m.group(1), m.group(2)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        text = body[start:end].strip()
        if text:
            attachments.append(Attachment(reg_number, name, ext, text))
    return attachments


def parse_file(path: Path) -> TenderDoc | None:
    raw = path.read_text(encoding="utf-8", errors="replace")
    parts = _SEPARATOR.split(raw, maxsplit=1)
    header_raw = parts[0]
    body = parts[1] if len(parts) > 1 else ""

    header = _parse_header(header_raw)
    reg_number = header.get("НОМЕР", path.stem)

    doc = TenderDoc(
        reg_number=reg_number,
        title=header.get("НАЗВАНИЕ", ""),
        customer=header.get("ЗАКАЗЧИК", ""),
        price=header.get("НМЦК", ""),
        source_path=str(path),
        attachments=_parse_sections(body, reg_number),
    )
    return doc


def load_corpus(docs_dir: Path) -> list[TenderDoc]:
    docs = []
    for path in sorted(docs_dir.glob("*.txt")):
        doc = parse_file(path)
        if doc and doc.attachments:
            docs.append(doc)
    return docs
