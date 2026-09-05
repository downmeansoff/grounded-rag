"""Строит синтетический шум для замера деградации поиска от размера корпуса.

Использование:
    python scripts/make_noise.py <папка_с_документами> <разметка> <куда> <сколько>
    python scripts/make_noise.py sample/docs eval/sample.json .noise 200

Четырнадцать документов это корпус для отладки, а не рабочий размер. Вопрос,
на который отвечает замер: что происходит с выдачей, когда у правильного чанка
становятся сотни похожих соседей. Ответы при этом остаются в исходных
документах, добавляются только чужие, поэтому меняется ровно одно, размер
корпуса.

Шум обязан быть тяжёлым, иначе замер соврёт в свою пользу. Документы из чужой
предметной области отличаются одной лексикой, вытеснять нужный чанк они не
будут, и кривая выйдет пологой просто потому, что шум легко отличить. Поэтому
куски берутся из самих исходных документов и перемешиваются между собой.

Куски с эталонными фразами выбрасываются: иначе ответ на вопрос оказался бы
сразу в нескольких документах, проверка разметки это заметила бы и уронила
прогон, и была бы права.

Режется по строкам, а не по абзацам. Выгрузка тендеров почти не разделена
пустыми строками: на настоящем корпусе из четырнадцати документов их всего 52
медианой в семь тысяч символов, и документы из такого материала вышли бы
повторением полусотни одинаковых кусков. Одинаковые конкуренты вытесняют хуже
разных, и замер снова соврал бы в свою пользу.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

PIECE_CHARS = 1200
SEED = 20260905

# Шапка в формате профиля tenders. Ключи без цифр: разбирает их выражение
# [А-ЯA-Z,() ]+, и «ОКПД2» превратилось бы в «ОКПД».
CUSTOMERS = (
    "ГБУЗ «ГОРОДСКАЯ БОЛЬНИЦА №{n}»",
    "МБОУ «СРЕДНЯЯ ШКОЛА №{n}»",
    "ФГБОУ ВО «ТЕХНИЧЕСКИЙ УНИВЕРСИТЕТ №{n}»",
    "МБУК «ДОМ КУЛЬТУРЫ №{n}»",
    "ГАУ «СПОРТИВНЫЙ КОМПЛЕКС №{n}»",
)
TITLES = (
    "Оказание услуг по гардеробному обслуживанию",
    "Услуги вахтера (сторожа)",
    "Оказание услуг по уборке помещений",
    "Комплексное обслуживание посетителей",
)


def gold_phrases(labeled_path: Path) -> list[str]:
    data = json.loads(labeled_path.read_text(encoding="utf-8"))
    questions = data["questions"] if isinstance(data, dict) else data
    return [gold["phrase"] for question in questions for gold in question["gold"]]


def donor_pieces(docs_dir: Path, gold: list[str]) -> list[str]:
    pieces: list[str] = []
    for path in sorted(docs_dir.glob("*.txt")):
        body = path.read_text(encoding="utf-8", errors="replace").split("=" * 70, 1)[-1]
        buffer: list[str] = []
        size = 0
        for line in body.splitlines():
            buffer.append(line)
            size += len(line) + 1
            if size < PIECE_CHARS:
                continue
            piece = "\n".join(buffer).strip()
            buffer, size = [], 0
            if len(piece) >= 200 and not any(phrase in piece for phrase in gold):
                pieces.append(piece)
    return pieces


def average_length(docs_dir: Path) -> int:
    """Шумовой документ должен быть той же длины, что настоящий.

    Иначе короткие документы дали бы меньше чанков на документ, и корпус из
    тысячи штук оказался бы легче настоящего при том же числе документов.
    """
    sizes = [len(p.read_text(encoding="utf-8", errors="replace")) for p in docs_dir.glob("*.txt")]
    return sum(sizes) // len(sizes) if sizes else 55_000


def main(docs_dir: Path, labeled_path: Path, out_dir: Path, count: int) -> None:
    gold = gold_phrases(labeled_path)
    pieces = donor_pieces(docs_dir, gold)
    if not pieces:
        print("не из чего собирать шум: в документах нет кусков без эталонных фраз")
        sys.exit(1)

    target = average_length(docs_dir)
    print(f"кусков-доноров: {len(pieces)}, длина документа: {target} символов")

    rng = random.Random(SEED)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        body: list[str] = []
        size = 0
        while size < target:
            piece = rng.choice(pieces)
            body.append(piece)
            size += len(piece) + 2
        header = (
            f"НОМЕР: 99{i:016d}\n"
            f"НАЗВАНИЕ: {rng.choice(TITLES)}\n"
            f"ЗАКАЗЧИК: {rng.choice(CUSTOMERS).format(n=rng.randint(1, 400))}\n"
            f"НМЦК: {rng.randrange(1_000, 9_000) * 1000}.00\n"
            f"СРОК, ДНЕЙ: {rng.randrange(90, 700)}\n"
            f"ФАЙЛЫ: описание объекта закупки.docx\n"
            + "=" * 70
            + "\n\n### Описание объекта закупки [docx]\n\n"
        )
        (out_dir / f"99{i:016d}.txt").write_text(header + "\n\n".join(body), encoding="utf-8")

    print(f"записано документов: {count} в {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print(
            "Использование: python scripts/make_noise.py "
            "<папка_с_документами> <разметка> <куда> <сколько>"
        )
        sys.exit(1)
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]))
