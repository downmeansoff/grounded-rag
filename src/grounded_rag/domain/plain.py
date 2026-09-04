"""Профиль без предметной области: каталог с .txt и .md, один файл один документ.

Нужен по двум причинам. Практическая: чтобы навести движок на папку с
документацией или конспектами, не описывая под неё формат. Проверочная:
абстракция с единственной реализацией ничего не доказывает, и пока рядом с
тендерами не встал профиль с другим разбором, другими метаданными и другими
промптами, «домен выбирается» оставалось бы заявлением.

Названием документа считается первая непустая строка: в markdown это заголовок,
в текстовом файле обычно тема. Если строка длинная, это скорее абзац, чем
заголовок, и тогда названием остаётся имя файла.
"""

from __future__ import annotations

from pathlib import Path

from grounded_rag.domain.base import Document, DomainProfile, Part

SUFFIXES = (".txt", ".md")
TITLE_MAX = 120


def _first_heading(raw: str) -> str:
    for line in raw.splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            return line if len(line) <= TITLE_MAX else ""
    return ""


def parse_file(path: Path) -> Document | None:
    raw = path.read_text(encoding="utf-8", errors="replace")
    if not raw.strip():
        return None
    return Document(
        doc_id=path.stem,
        title=_first_heading(raw) or path.stem,
        source_path=str(path),
        meta={"Файл": path.name},
        # Файл не делится на части, но чанки всё равно должны знать, откуда
        # они: имя файла и есть единственная часть документа.
        parts=[Part(name=path.name, text=raw, ext=path.suffix.lstrip("."))],
        raw=raw,
    )


class PlainProfile(DomainProfile):
    name = "plain"
    entity = "документ"
    corpus = "документам корпуса"
    prompt_version = "plain-1"

    def load(self, docs_dir: Path) -> list[Document]:
        docs = []
        for path in sorted(docs_dir.iterdir()):
            if path.suffix.lower() not in SUFFIXES:
                continue
            doc = parse_file(path)
            if doc:
                docs.append(doc)
        return docs

    @property
    def context_system(self) -> str:
        return (
            "Ты пишешь короткие пояснения к фрагментам документов, чтобы их лучше "
            "находил поиск. Ответ - одно-два законченных предложения, в которых "
            "названы тема документа и раздел, откуда взят фрагмент. Не отвечай одним "
            "словом и не отвечай заголовком раздела. Не пересказывай сам фрагмент и "
            "не добавляй вступлений.\n"
            "Пример ответа: «Фрагмент из раздела про установку в руководстве по "
            "развёртыванию сервиса: перечисляет требования к версии Python.»"
        )

    def context_prompt(self, doc: Document, part_name: str, head: str, text: str) -> str:
        return (
            f"Документ: {doc.title}\n"
            f"Файл: {part_name}\n\n"
            f"Начало документа:\n{head}\n\n"
            f"Фрагмент:\n{text}\n\n"
            "Напиши, к чему относится этот фрагмент внутри документа."
        )
