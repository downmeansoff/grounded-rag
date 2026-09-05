"""Третий профиль: база знаний в markdown.

Проверяется то, ради чего он заведён, а не разбор markdown как таковой. Первые
два профиля делят документ плоско (шаблон тендера и файл целиком), этот держит
структуру: разделы становятся частями, frontmatter метаданными. Значит, здесь
впервые появляются два места, где движок может соврать молча, - имя части,
которым чанк ссылается на своё место, и метаданные, ключей которых движок не
знает заранее.
"""

from __future__ import annotations

import pytest

from grounded_rag.config import settings
from grounded_rag.contextual.cache import cache_key
from grounded_rag.domain.factory import make_domain
from grounded_rag.domain.markdown import PREAMBLE, MarkdownProfile, parse_file
from grounded_rag.domain.plain import PlainProfile

NOTE = """---
id: deploy-guide
title: Развёртывание сервиса
автор: Глеб
статус: черновик
теги:
  - инфраструктура
---

# Развёртывание сервиса

Заметка о том, как поднять сервис на новой машине.

## Требования

Python 3.11 и доступ к базе.

### Версии библиотек

psycopg 3 и pgvector.

## Откат

Снести контейнер и накатить прошлый образ.
"""


def _write(tmp_path, name: str, text: str):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_frontmatter_gives_identity_and_metadata(tmp_path):
    doc = parse_file(_write(tmp_path, "note.md", NOTE))

    assert doc.doc_id == "deploy-guide"  # не имя файла: id задан явно
    assert doc.title == "Развёртывание сервиса"
    # id и title ушли в свои поля, остальное осталось метаданными как есть.
    assert doc.meta == {"автор": "Глеб", "статус": "черновик"}


def test_list_values_do_not_leak_into_metadata(tmp_path):
    # meta это словарь строк и уезжает в JSONB. Список тегов в него не влезает,
    # и «теги: » с пустым значением там тоже не нужен.
    doc = parse_file(_write(tmp_path, "note.md", NOTE))
    assert "теги" not in doc.meta
    assert "инфраструктура" not in doc.meta.values()


def test_sections_become_parts_and_subheadings_stay_inside(tmp_path):
    doc = parse_file(_write(tmp_path, "note.md", NOTE))

    assert [p.name for p in doc.parts] == [PREAMBLE, "Требования", "Откат"]
    # Подзаголовок ### это часть своего раздела, а не отдельная часть.
    assert "psycopg 3 и pgvector." in doc.part("Требования").text
    assert "Откат" not in doc.part("Требования").text


def test_title_is_not_repeated_in_the_text(tmp_path):
    # Заголовок документа уже стал title и попадёт в цитату оттуда.
    doc = parse_file(_write(tmp_path, "note.md", NOTE))
    assert doc.part(PREAMBLE).text == "Заметка о том, как поднять сервис на новой машине."


def test_repeated_section_names_are_numbered(tmp_path):
    # Имя части это адрес чанка: по нему собирается цитата и по нему
    # контекстуализатор ищет шапку раздела. Два раздела с одним именем сделали
    # бы половину ссылок неверными, и ошибка была бы невидимой.
    doc = parse_file(
        _write(
            tmp_path,
            "dup.md",
            "# Заметка\n\n## Установка\n\nЧерез pip.\n\n## Установка\n\nИз исходников.\n",
        )
    )

    assert [p.name for p in doc.parts] == ["Установка", "Установка (2)"]
    assert doc.part("Установка").text == "Через pip."
    assert doc.part("Установка (2)").text == "Из исходников."


def test_note_without_sections_is_taken_whole(tmp_path):
    doc = parse_file(_write(tmp_path, "flat.md", "# Мысль\n\nОдин абзац без разделов.\n"))

    assert doc.doc_id == "flat"  # frontmatter нет, идентификатор из имени файла
    assert doc.title == "Мысль"
    assert [(p.name, p.text) for p in doc.parts] == [(PREAMBLE, "Один абзац без разделов.")]


def test_note_with_nothing_but_a_title_is_skipped(tmp_path):
    # Индексировать нечего: заголовок уже лежит в documents.title.
    assert parse_file(_write(tmp_path, "empty.md", "# Только заголовок\n")) is None


def test_load_walks_subfolders_and_ignores_other_formats(tmp_path):
    _write(tmp_path, "a.md", "# A\n\nПервая.\n")
    _write(tmp_path, "sub/b.markdown", "# B\n\nВторая.\n")
    _write(tmp_path, "sub/c.txt", "Не markdown.")
    _write(tmp_path, "d.md", "")

    docs = MarkdownProfile().load(tmp_path)

    assert [d.doc_id for d in docs] == ["a", "b"]


def test_load_records_the_path_relative_to_the_corpus(tmp_path):
    # Путь это ключ фильтра ("Файл=sub/"), поэтому он относительный: абсолютный
    # менялся бы от машины к машине, а заметка без frontmatter иначе осталась бы
    # вовсе без метаданных и недостижимой ни одним фильтром.
    _write(tmp_path, "sub/b.markdown", "# B\n\nВторая.\n")
    _write(tmp_path, "note.md", NOTE)

    by_id = {doc.doc_id: doc for doc in MarkdownProfile().load(tmp_path)}

    assert by_id["b"].meta == {"Файл": "sub/b.markdown"}
    assert by_id["deploy-guide"].meta["Файл"] == "note.md"
    assert by_id["deploy-guide"].meta["автор"] == "Глеб"  # своё не перетёрто


def test_citation_names_the_section(tmp_path):
    profile = MarkdownProfile()
    assert profile.citation("deploy-guide", "Требования", 2) == "заметка deploy-guide, Требования#2"


def test_context_prompt_carries_metadata_of_the_note(tmp_path):
    profile = MarkdownProfile()
    doc = parse_file(_write(tmp_path, "note.md", NOTE))

    prompt = profile.context_prompt(doc, "Требования", "Python 3.11", "psycopg 3 и pgvector.")

    assert "Развёртывание сервиса" in prompt
    assert "статус: черновик" in prompt
    assert "Раздел: Требования" in prompt


def test_profile_is_reachable_by_settings():
    assert isinstance(make_domain(settings.model_copy(update={"domain": "markdown"})), MarkdownProfile)


def test_prompt_version_separates_the_cache_from_other_profiles():
    # Один и тот же .md-файл plain и markdown описывают моделью по-разному, и
    # цена ошибки тут денежная: совпади версии, чужой контекст молча подставился
    # бы вместо своего.
    markdown, plain = MarkdownProfile(), PlainProfile()
    assert markdown.prompt_version != plain.prompt_version

    args = ("note", "Требования", "psycopg 3 и pgvector.")
    assert cache_key(markdown.prompt_version, *args) != cache_key(plain.prompt_version, *args)


@pytest.mark.parametrize("suffix", [".md", ".markdown"])
def test_part_keeps_the_extension_of_its_file(tmp_path, suffix):
    doc = parse_file(_write(tmp_path, f"note{suffix}", "# Заметка\n\n## Раздел\n\nТекст.\n"))
    assert doc.parts[0].ext == suffix.lstrip(".")
