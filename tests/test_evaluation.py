"""Метрики и разметка: считаются без базы и без модели.

Проверять здесь надо не качество поиска, а честность линейки. Если hit@5
округляет в свою пользу или разметка молча принимает фрагмент, которого в
документе нет, то любое число из замера перестаёт что-либо значить.
"""

from __future__ import annotations

from grounded_rag.domain.plain import PlainProfile
from grounded_rag.evaluation.dataset import check_against_corpus, check_against_index
from grounded_rag.evaluation.metrics import (
    Gold,
    Question,
    hit_at_k,
    normalize,
    recall_at_k,
    reciprocal_rank,
    relevance,
)

GOLD = (Gold("111", "влажную уборку помещений"),)


def test_whitespace_does_not_break_a_match():
    # Таблицы в документах приезжают с рваными отступами, и эталон
    # отличается от чанка только ими. Считать это промахом поиска нельзя.
    chunk = "Исполнитель  обеспечивает\n влажную   уборку\tпомещений гардеробов"
    assert relevance([chunk], GOLD) == [True]


def test_case_is_significant():
    # Регистр в этих текстах несёт смысл: НМЦК, ОКПД2, Заказчик как сторона
    # договора. Схлопывать его значит ловить лишние совпадения.
    assert relevance(["ВЛАЖНУЮ УБОРКУ ПОМЕЩЕНИЙ"], GOLD) == [False]


def test_normalize_collapses_and_trims():
    assert normalize("  a \n\t b  ") == "a b"


def test_hit_looks_only_at_top_k():
    flags = [False, False, False, False, False, True]
    assert hit_at_k(flags, 5) is False
    assert hit_at_k(flags, 6) is True


def test_reciprocal_rank_is_the_first_hit():
    assert reciprocal_rank([False, True, True]) == 0.5
    assert reciprocal_rank([True]) == 1.0


def test_reciprocal_rank_is_zero_when_nothing_found():
    assert reciprocal_rank([False, False]) == 0.0


def test_recall_counts_gold_fragments_not_positions():
    # На вопрос отвечают три документа, поиск принёс два. Пять занятых мест в
    # выдаче этого не исправляют: треть ответа осталась ненайденной.
    gold = (Gold("1", "первый"), Gold("2", "второй"), Gold("3", "третий"))
    texts = ["текст первый", "и первый снова", "второй", "мимо", "мимо"]
    assert recall_at_k(texts, gold, 5) == 2 / 3


def test_recall_ignores_what_lies_below_k():
    gold = (Gold("1", "первый"), Gold("2", "второй"))
    texts = ["первый", "мимо", "второй"]
    assert recall_at_k(texts, gold, 2) == 0.5


def test_corpus_check_catches_a_fragment_from_another_document(tmp_path):
    # Самая дорогая ошибка разметки: идентификатор документа проставлен наугад.
    # Замер тогда показывает ноль там, где поиск отработал верно.
    (tmp_path / "111.txt").write_text("влажную уборку помещений", encoding="utf-8")
    (tmp_path / "222.txt").write_text("ничего похожего", encoding="utf-8")

    question = Question("q", "уборка", (Gold("222", "влажную уборку помещений"),))
    problems = check_against_corpus([question], tmp_path, PlainProfile())

    assert len(problems) == 2
    assert "нет в 222" in problems[0]
    assert "встречается ещё в 111" in problems[1]


def test_corpus_check_accepts_a_fragment_listed_in_every_document_that_has_it(tmp_path):
    (tmp_path / "111.txt").write_text("влажную уборку помещений", encoding="utf-8")
    (tmp_path / "222.txt").write_text("влажную   уборку\nпомещений", encoding="utf-8")

    question = Question(
        "q",
        "уборка",
        (Gold("111", "влажную уборку помещений"), Gold("222", "влажную уборку помещений")),
    )
    assert check_against_corpus([question], tmp_path, PlainProfile()) == []


def test_index_check_catches_a_fragment_split_across_chunks():
    # Проверки по документу тут мало: фраза в документе есть, а искать поиск
    # умеет только чанки целиком, и ни один её не содержит.
    chunks = [("111", 0, "Исполнитель обеспечивает влажную"), ("111", 1, "уборку помещений")]
    question = Question("q", "уборка", (Gold("111", "влажную уборку помещений"),))

    problems = check_against_index([question], chunks)

    assert len(problems) == 1
    assert "ни в одном его чанке" in problems[0]


def test_index_check_passes_when_a_chunk_holds_the_whole_fragment():
    chunks = [("111", 0, "мимо"), ("111", 1, "обеспечивает влажную   уборку\nпомещений гардеробов")]
    question = Question("q", "уборка", (Gold("111", "влажную уборку помещений"),))

    assert check_against_index([question], chunks) == []


def test_index_check_ignores_a_chunk_of_another_document():
    # Фраза нашлась, но в чужом документе. Для разметки это не ответ.
    chunks = [("222", 0, "влажную уборку помещений")]
    question = Question("q", "уборка", (Gold("111", "влажную уборку помещений"),))

    assert len(check_against_index([question], chunks)) == 1
