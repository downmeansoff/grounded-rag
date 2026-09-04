"""Чанкер: размер, overlap, индексы.

Свойства, а не золотые строки: конкретные тексты корпуса меняются,
инвариант «ни один чанк не длиннее size» ломаться не должен никогда.
"""

from __future__ import annotations

from grounded_rag.chunk.recursive import chunk_text


def test_empty_text_gives_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("   \n\t  ") == []


def test_short_text_stays_one_chunk():
    chunks = chunk_text("Короткий текст про гардероб.", size=1500)
    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert chunks[0].text == "Короткий текст про гардероб."


def test_no_chunk_exceeds_size():
    text = "\n\n".join(f"Пункт {i}. " + "слово " * 60 for i in range(40))
    for chunk in chunk_text(text, size=1500, overlap=200):
        assert len(chunk.text) <= 1500


def test_indexes_are_sequential_from_zero():
    text = "\n\n".join("условия оказания услуг " * 30 for _ in range(20))
    chunks = chunk_text(text)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_long_text_splits_into_several_chunks():
    text = "\n\n".join(f"Пункт {i}: условия оказания услуг. " * 10 for i in range(30))
    assert len(chunk_text(text, size=1500)) > 1


def test_overlap_repeats_tail_of_previous_chunk():
    # Ни абзацев, ни пробелов — режется прямо по символам, overlap виден буквально.
    text = "абвгдежзий" * 500
    chunks = chunk_text(text, size=1000, overlap=100)
    assert len(chunks) > 1
    assert chunks[1].text.startswith(chunks[0].text[-100:])


def test_spaces_and_tabs_collapsed():
    chunks = chunk_text("много      пробелов\tи\tтабов")
    assert chunks[0].text == "много пробелов и табов"


def test_paragraph_boundary_preferred_over_mid_word_cut():
    # Два абзаца, вместе не влезают в size — резать должно ровно по границе абзацев.
    first = "Первый абзац. " * 40
    second = "Второй абзац. " * 40
    chunks = chunk_text(f"{first}\n\n{second}", size=700, overlap=0)
    assert chunks[0].text.endswith("Первый абзац.")
