#!/usr/bin/env python3
"""Собрать docx из main.tex для правки в текстовом редакторе.

Макросы стиля qwen.sty (\\remeasure, \\doubt, \\qtable, \\thead и прочие)
pandoc не знает, потому что они объявлены во внешнем пакете. Скрипт
раскрывает их в обычный LaTeX, пишет промежуточный файл и вызывает pandoc.

    python3 tex2docx.py [--keep]

Сложность: O(n) по длине файла, все замены однопроходные.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "main.tex"
INTERMEDIATE = HERE / "main.export.tex"
TARGET = HERE / "main.docx"

# Преамбула промежуточного файла: те же макросы, но в терминах, которые
# читает pandoc. Цвет задаётся через \textcolor, он становится цветным
# текстом в docx.
PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[T2A]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[russian]{babel}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{hyperref}
\newcommand{\remeasure}[1]{\textcolor{red}{(перемерить: #1)}}
\newcommand{\doubt}[1]{\textcolor{red}{(вероятно неверно: #1)}}
\newcommand{\lead}[1]{\textbf{#1}}
\newcommand{\thead}[1]{\textbf{#1}}
\newcommand{\tstub}[1]{#1}
\newcommand{\srcnote}[1]{ Источник: \texttt{#1}.}
\newcommand{\hairline}{}
\newcommand{\displayfamily}{}
"""


def strip_preamble(text: str) -> str:
    """Вернуть тело документа без преамбулы исходного файла."""
    start = text.index(r"\begin{document}")
    return text[start + len(r"\begin{document}"):]


def read_group(text: str, start: int) -> tuple[str, int]:
    """Прочитать группу в фигурных скобках со вложенностью.

    Возвращает содержимое без внешних скобок и позицию после группы.
    Регулярное выражение здесь не годится: спецификация столбцов вида
    N{4}{1} сама содержит скобки.
    """
    if text[start] != "{":
        raise ValueError(f"ожидалась группа в позиции {start}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1:i], i + 1
    raise ValueError("группа не закрыта")


def convert_qtables(text: str) -> str:
    """Развернуть окружение qtable в table и tabular.

    Спецификация столбцов qtable содержит типы N{целых}{дробных} и
    P{ширина}, которых в обычном LaTeX нет: первый становится правым
    выравниванием, второй левым.
    """
    def column_spec(spec: str) -> str:
        spec = re.sub(r"N\{\d+\}\{\d+\}", "r", spec)
        spec = re.sub(r"P\{[^}]*\}", "l", spec)
        return spec

    opener = r"\begin{qtable}"
    closer = r"\end{qtable}"
    out = []
    pos = 0
    while True:
        begin = text.find(opener, pos)
        if begin < 0:
            out.append(text[pos:])
            return "".join(out)
        out.append(text[pos:begin])

        i = begin + len(opener)
        if i < len(text) and text[i] == "[":          # необязательное размещение
            i = text.index("]", i) + 1
        spec, i = read_group(text, i)
        caption, i = read_group(text, i)
        label, i = read_group(text, i)
        end = text.index(closer, i)
        body = text[i:end].strip()

        out.append(
            "\\begin{table}[htbp]\n\\centering\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
            f"\\begin{{tabular}}{{{column_spec(spec)}}}\n\\toprule\n"
            f"{body}\n"
            "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
        )
        pos = end + len(closer)


def decimal_commas(text: str) -> str:
    """Поставить запятую как десятичный разделитель в строках таблиц.

    В исходнике числа записаны с точкой, потому что их форматирует siunitx.
    Замена применяется только к строкам с разделителем столбцов, поэтому
    имена файлов и версии в обычном тексте не затрагиваются.
    """
    lines = []
    for line in text.split("\n"):
        if "&" in line:
            line = re.sub(r"(?<=\d)\.(?=\d)", ",", line)
        lines.append(line)
    return "\n".join(lines)


MARKERS = ("перемерить:", "вероятно неверно:")
RED = "C00000"


def colorize_marks(path: Path) -> int:
    """Покрасить пометки о перемере в красный прямо в готовом docx.

    Команда \\textcolor теряется при записи docx, поэтому цвет
    возвращается на уровне разметки: каждый фрагмент текста с меткой
    получает свойство цвета. Возвращает число покрашенных фрагментов.
    """
    document = "word/document.xml"
    with zipfile.ZipFile(path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}

    xml = entries[document].decode("utf-8")
    painted = 0

    def paint(match: re.Match) -> str:
        nonlocal painted
        inner = match.group(1)
        if not any(marker in inner for marker in MARKERS):
            return match.group(0)
        painted += 1
        color = f'<w:color w:val="{RED}" />'
        if inner.startswith("<w:rPr>"):
            # Порядок элементов внутри свойств фрагмента задан схемой:
            # цвет идёт после начертания, поэтому вставка в конец группы.
            return "<w:r>" + inner.replace("</w:rPr>", color + "</w:rPr>", 1) + "</w:r>"
        return f"<w:r><w:rPr>{color}</w:rPr>{inner}</w:r>"

    xml = re.sub(r"<w:r>(.*?)</w:r>", paint, xml, flags=re.DOTALL)
    entries[document] = xml.encode("utf-8")

    temporary = path.with_suffix(".docx.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    shutil.move(temporary, path)
    return painted


def build(keep: bool) -> int:
    text = SOURCE.read_text(encoding="utf-8")
    body = strip_preamble(text)

    body = convert_qtables(body)
    body = decimal_commas(body)

    # Врезка становится цитатой: у pandoc нет нашего окружения с линейкой.
    body = body.replace(r"\begin{note}", r"\begin{quote}")
    body = body.replace(r"\end{note}", r"\end{quote}")

    # Группа вокруг запятой нужна только для интервалов в LaTeX.
    body = body.replace("{,}", ",")

    # Заголовочный блок исходника несёт команды оформления, которые в docx
    # не нужны; вместо него простой титул.
    body = re.sub(r"\\maketitle", "", body)
    title = (
        "\\title{Qwen3.5 на Android: GPU и NPU\\\\"
        "Итоги работы 17--18 сентября 2026 года}\n"
        "\\author{Григорий Евко}\n\\date{}\n"
    )

    INTERMEDIATE.write_text(
        PREAMBLE + title + "\\begin{document}\n\\maketitle\n" + body,
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "pandoc",
            str(INTERMEDIATE),
            "--from=latex",
            "--to=docx",
            "--output=" + str(TARGET),
            "--standalone",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        return result.returncode
    if result.stderr.strip():
        sys.stderr.write(result.stderr)

    if not keep:
        INTERMEDIATE.unlink(missing_ok=True)

    painted = colorize_marks(TARGET)
    print(f"{TARGET} записан, {TARGET.stat().st_size} байт, "
          f"пометок покрашено: {painted}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep",
        action="store_true",
        help="оставить промежуточный main.export.tex для разбора ошибок",
    )
    raise SystemExit(build(parser.parse_args().keep))
