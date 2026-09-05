"""Заказчик из текста запроса в фильтр по метаданным.

Фильтр умеет то, чего не умеет поиск по тексту: отсечь чужие закупки до отбора
кандидатов. Но задавать его приходилось руками, хотя в самом вопросе имя
заказчика обычно уже есть ("какой штраф в тюменской поликлинике"). Здесь оно
оттуда достаётся.

Сравниваются не подстроки, а лексемы русского словаря Postgres. Подстрока не
срабатывает ни в одну сторону: в вопросе "тюменской поликлинике", в
метаданных "ТЮМЕНСКОЙ ОБЛАСТИ ... ГОРОДСКАЯ ПОЛИКЛИНИКА №5" - разные падежи и
разный порядок слов. to_tsvector('russian') приводит и то и другое к основам и
заодно выбрасывает предлоги. Того же словаря, что и полнотекстовая часть
поиска: две разные нормализации в одном движке рано или поздно разъезжаются.

Лексема идёт в счёт, только если проходит два отсева.

Первый: она встречается ровно у одного заказчика. "Государственное",
"бюджетное", "учреждение" есть почти у всех и заказчиков не различают, а со
счётом по ним побеждал бы тот, у кого длиннее название.

Второй: она редка в тексте самого корпуса. Первого отсева мало, и видно это на
живом наборе: "медицинский" стоит в названии ровно одного заказчика, но в
текстах закупок встречается в 9 документах из 14 - это слово предметной
области, а не имя. Вопрос "нужны ли гардеробщикам медицинские книжки" из-за
него уезжал в медицинский университет, где ответа нет. У настоящих имён
("тюменский", "мордовский", "шолом-алейхем") частота по корпусу нулевая или
единичная, так что порог разделяет их с большим запасом.

Побеждает заказчик с наибольшим числом совпавших лексем, прошедших оба отсева.
Ноль совпадений или ничья означают, что фильтра нет: искать по всему корпусу
хуже, чем найти сразу, но лучше, чем сузить корпус до чужого заказчика и
получить уверенный ответ не из того документа.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Collection, Mapping

import psycopg

# Доля документов, после которой лексема считается словом предметной области, а
# не именем. Четверть корпуса: на живом наборе имена заказчиков стоят в одном
# документе из четырнадцати, общие слова - в девяти и больше, и порог проходит
# посередине пустого промежутка, а не по краю распределения.
COMMON_SHARE = 0.25


def _lexemes(conn: psycopg.Connection, text: str) -> set[str]:
    rows = conn.execute("SELECT lexeme FROM unnest(to_tsvector('russian', %s))", (text,)).fetchall()
    return {row[0] for row in rows}


def customer_lexemes(conn: psycopg.Connection, key: str) -> dict[str, set[str]]:
    """Лексемы названия каждого заказчика в индексе, одним запросом.

    Одним, потому что замер гоняет полсотни вопросов подряд: список заказчиков
    от вопроса к вопросу не меняется, и читать его каждый раз заново значит
    сделать полсотни одинаковых чтений вместо одного.
    """
    rows = conn.execute(
        """
        SELECT names.value, lexemes.lexeme
        FROM (
            SELECT DISTINCT d.meta ->> %(key)s AS value
            FROM documents d
            WHERE COALESCE(d.meta ->> %(key)s, '') <> ''
        ) AS names,
        LATERAL unnest(to_tsvector('russian', names.value)) AS lexemes
        """,
        {"key": key},
    ).fetchall()

    by_customer: dict[str, set[str]] = {}
    for value, lexeme in rows:
        by_customer.setdefault(value, set()).add(lexeme)
    return by_customer


def common_lexemes(
    conn: psycopg.Connection,
    by_customer: Mapping[str, set[str]],
    share: float = COMMON_SHARE,
) -> set[str]:
    """Лексемы имён, которые корпус использует как обычные слова.

    Считаются только лексемы из имён заказчиков: остальной словарь корпуса на
    выбор не влияет, а считать его целиком значит читать весь индекс ради
    чисел, которые никто не спросит.
    """
    wanted = sorted({lexeme for lexemes in by_customer.values() for lexeme in lexemes})
    if not wanted:
        return set()

    total = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    if not total:
        return set()

    rows = conn.execute(
        """
        SELECT lexemes.lexeme, COUNT(DISTINCT c.doc_id) AS documents
        FROM chunks c, LATERAL unnest(c.tsv) AS lexemes
        WHERE lexemes.lexeme = ANY(%(wanted)s)
        GROUP BY lexemes.lexeme
        """,
        {"wanted": wanted},
    ).fetchall()
    return {lexeme for lexeme, documents in rows if documents > share * total}


def pick_customer(
    query_lexemes: Collection[str],
    by_customer: Mapping[str, set[str]],
    common: Collection[str] = (),
) -> str | None:
    """Заказчик, чьи различающие лексемы встретились в запросе.

    Чистая функция: вся работа с базой снаружи, поэтому правило отбора можно
    проверить тестом без индекса и без корпуса.
    """
    owners = Counter(lexeme for lexemes in by_customer.values() for lexeme in lexemes)
    wanted = set(query_lexemes)
    ignored = set(common)

    scores = {
        value: sum(
            1 for lexeme in lexemes if owners[lexeme] == 1 and lexeme not in ignored and lexeme in wanted
        )
        for value, lexemes in by_customer.items()
    }

    best = max(scores.values(), default=0)
    if best == 0:
        return None

    winners = [value for value, score in scores.items() if score == best]
    return winners[0] if len(winners) == 1 else None


def auto_filter(
    conn: psycopg.Connection,
    query_text: str,
    key: str,
    filters: Mapping[str, str] | None = None,
    known: Mapping[str, set[str]] | None = None,
    common: Collection[str] | None = None,
) -> tuple[dict[str, str], str | None]:
    """Фильтры для retrieve и найденный заказчик (None, если не нашёлся).

    Заданный руками фильтр по тому же ключу не трогается: явное условие всегда
    сильнее угаданного, иначе опечатка в имени заказчика молча заменялась бы
    догадкой движка.

    known и common это те же два справочника, посчитанные заранее: за прогон
    замера они не меняются, а вопросов полсотни.
    """
    merged = dict(filters or {})
    if not key or key in merged:
        return merged, None

    by_customer = customer_lexemes(conn, key) if known is None else known
    ignored = common_lexemes(conn, by_customer) if common is None else common
    found = pick_customer(_lexemes(conn, query_text), by_customer, ignored)
    if found:
        merged[key] = found
    return merged, found


QUOTED = re.compile(r'[«"\'](.+?)[»"\']', re.DOTALL)


def short_name(value: str, limit: int = 40) -> str:
    """Различающая часть имени заказчика, для строк отчёта.

    Обрезка по первым сорока символам для таких имён бесполезна: «МУНИЦИПАЛЬНОЕ
    БЮДЖЕТНОЕ УЧРЕЖДЕНИЕ КУЛЬТУРЫ ...» и «МУНИЦИПАЛЬНОЕ БЮДЖЕТНОЕ УЧРЕЖДЕНИЕ
    ЗДРАВООХРАНЕНИЯ ...» в сорок символов дают одно и то же начало, и по строке
    замера не видно, какого заказчика движок выбрал. Различает их то, что стоит
    в кавычках, поэтому берётся оно.
    """
    quoted = QUOTED.search(value)
    name = quoted.group(1).strip() if quoted else value.strip()
    return name if len(name) <= limit else name[: limit - 1] + "…"


def explain(domain_name: str, key: str, found: str | None) -> str:
    """Строка про то, что автофильтр сделал, одна на все скрипты.

    Молчать нельзя: разница между "заказчик опознан" и "ищу по всему корпусу"
    объясняет выдачу, а по самой выдаче её не видно.
    """
    if not key:
        return f'Автофильтр: профиль "{domain_name}" не хранит имя заказчика в метаданных, фильтр не подставлен.'
    if found is None:
        return "Автофильтр: заказчик в запросе не опознан, ищу по всему корпусу."
    return f"Автофильтр: {key}={found}"
