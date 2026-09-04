"""Профиль для базы знаний в markdown: заметки с frontmatter и разделами.

Отличие от plain не в расширении файла, а в том, что документ здесь имеет
структуру, и структуру эту стоит сохранить. Заметка про развёртывание сервиса
состоит из «Требований», «Установки» и «Отката», и фрагмент про версию Python
осмысленно цитировать как раздел «Требования», а не как файл целиком. Разделы
верхнего уровня становятся частями документа, и цитата получает имя раздела.

Метаданные берутся из YAML frontmatter, который в базах знаний вроде Obsidian
и в генераторах статических сайтов лежит в начале файла между строками ---.
Разбирается он построчно, без зависимости на парсер YAML: нужны только пары
ключ-значение верхнего уровня, а вложенные структуры и списки в метаданные
документа всё равно не годятся, потому что meta это словарь строк.

Третий профиль нужен ещё и как проверка самой абстракции. Тендеры и plain
делят документ на части по-разному, но одинаково плоско: один разбор по
шаблону и один файл целиком. Разделы markdown это первый случай, когда части
документа приходится нумеровать при совпадении имён и когда у документа есть
метаданные, которых движок не знает заранее.
"""

from __future__ import annotations

import re
from pathlib import Path

from grounded_rag.domain.base import Document, DomainProfile, Part

SUFFIXES = (".md", ".markdown")
_FRONTMATTER = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n?", re.DOTALL)
_H1 = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_SECTION = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
# Текст до первого раздела: обычно вводный абзац под заголовком документа.
PREAMBLE = "Введение"


def _parse_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    match = _FRONTMATTER.match(raw)
    if not match:
        return {}, raw

    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        key, sep, value = line.partition(":")
        # Вложенность (строка начинается с отступа) и списки пропускаются:
        # значением метаданных может быть только строка.
        if not sep or line[:1].isspace() or key.strip().startswith("-"):
            continue
        value = value.strip().strip('"').strip("'")
        if value:
            meta[key.strip()] = value
    return meta, raw[match.end():]


def _split_sections(body: str) -> list[tuple[str, str]]:
    """Разделы верхнего уровня. Подзаголовки остаются внутри своего раздела."""
    matches = list(_SECTION.finditer(body))
    if not matches:
        # Заголовок документа выкидывается из текста части: он уже стал title,
        # и повторять его в тексте чанка значит удваивать его в цитате.
        whole = _H1.sub("", body).strip()
        return [(PREAMBLE, whole)] if whole else []

    sections = []
    preamble = _H1.sub("", body[: matches[0].start()]).strip()
    if preamble:
        sections.append((PREAMBLE, preamble))

    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        text = body[match.end() : end].strip()
        if text:
            sections.append((match.group(1), text))
    return sections


def _unique(names: list[str]) -> list[str]:
    """Имена частей внутри документа обязаны различаться.

    Имя части это то, чем чанк ссылается на своё место: по нему собирается
    цитата и по нему же контекстуализатор ищет шапку раздела. Два раздела
    «Установка» в одном файле сделали бы половину этих ссылок неверными молча.
    """
    seen: dict[str, int] = {}
    result = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        result.append(name if seen[name] == 1 else f"{name} ({seen[name]})")
    return result


def parse_file(path: Path) -> Document | None:
    raw = path.read_text(encoding="utf-8", errors="replace")
    meta, body = _parse_frontmatter(raw)
    sections = _split_sections(body)
    if not sections:
        return None

    names = _unique([name for name, _ in sections])
    heading = _H1.search(body)
    return Document(
        doc_id=meta.pop("id", "") or path.stem,
        title=meta.pop("title", "") or (heading.group(1).strip() if heading else path.stem),
        source_path=str(path),
        meta=meta,
        parts=[
            Part(name=name, text=text, ext=path.suffix.lstrip("."))
            for name, (_, text) in zip(names, sections)
        ],
        raw=raw,
    )


class MarkdownProfile(DomainProfile):
    name = "markdown"
    entity = "заметка"
    corpus = "заметкам базы знаний"
    prompt_version = "markdown-1"

    def load(self, docs_dir: Path) -> list[Document]:
        docs = []
        for path in sorted(docs_dir.rglob("*")):
            if path.suffix.lower() not in SUFFIXES or not path.is_file():
                continue
            doc = parse_file(path)
            if doc:
                docs.append(doc)
        return docs

    @property
    def context_system(self) -> str:
        return (
            "Ты пишешь короткие пояснения к фрагментам заметок, чтобы их лучше находил "
            "поиск. Ответ - одно-два законченных предложения, в которых названы тема "
            "заметки и о чём идёт речь в разделе. Не отвечай одним словом и не отвечай "
            "заголовком раздела. Не пересказывай сам фрагмент и не добавляй вступлений.\n"
            "Пример ответа: «Фрагмент из раздела «Установка» заметки про развёртывание "
            "сервиса: перечисляет требования к версии Python и доступу к базе.»"
        )

    def context_prompt(self, doc: Document, part_name: str, head: str, text: str) -> str:
        meta = "".join(f"{key}: {value}\n" for key, value in doc.meta.items())
        return (
            f"Заметка: {doc.title}\n"
            f"{meta}"
            f"Раздел: {part_name}\n\n"
            f"Начало раздела:\n{head}\n\n"
            f"Фрагмент:\n{text}\n\n"
            "Напиши, к чему относится этот фрагмент внутри заметки."
        )
