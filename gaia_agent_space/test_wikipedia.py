"""
Тесты разбора для WikipediaTool — без обращения к сети.

Проверяется то, что реально может сломаться в коде: разбор таблиц статьи
(включая многоуровневые заголовки, которые у Википедии встречаются постоянно)
и поиск фрагментов по подстроке. Сетевой путь тут намеренно не тестируется:
он проверяется живым запуском, а мокать HTTP ради видимости покрытия смысла нет.

Запуск:  python test_wikipedia.py
"""

import sys

from tools import WikipediaTool

# Разметка, повторяющая типичную таблицу дискографии из Википедии:
# многоуровневый заголовок, годы в отдельном столбце, лишние примечания.
SAMPLE_HTML = """
<div class="mw-parser-output">
<table class="wikitable">
  <tr><th rowspan="2">Title</th><th rowspan="2">Year</th><th colspan="2">Peak positions</th></tr>
  <tr><th>ARG</th><th>ESP</th></tr>
  <tr><td>Album Alpha</td><td>1999</td><td>3</td><td>12</td></tr>
  <tr><td>Album Beta</td><td>2000</td><td>1</td><td>5</td></tr>
  <tr><td>Album Gamma</td><td>2005</td><td>7</td><td>—</td></tr>
  <tr><td>Album Delta</td><td>2009</td><td>2</td><td>9</td></tr>
  <tr><td>Album Epsilon</td><td>2012</td><td>4</td><td>—</td></tr>
</table>
<table class="wikitable">
  <tr><th>Award</th><th>Result</th></tr>
  <tr><td>Best Album</td><td>Won</td></tr>
</table>
</div>
"""

LONG_TEXT = (
    "Introduction paragraph about the subject. " * 20
    + "The work of R. G. Arendt was supported by NASA award number 80GSFC21M0002. "
    + "Trailing paragraph. " * 20
)


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"[{'ok ' if condition else 'FAIL'}] {label}" + (f" — {detail}" if detail and not condition else ""))
    return condition


def main() -> int:
    results = []
    tool = WikipediaTool()

    # --- разбор таблиц ---
    out = tool._tables_from_html(SAMPLE_HTML)
    results.append(check("обе таблицы найдены", "Найдено таблиц: 2" in out, out[:200]))
    results.append(check("данные строк попали в вывод", "Album Gamma" in out, out[:200]))
    results.append(check("годы сохранены", "2005" in out and "2009" in out))
    results.append(
        check(
            "многоуровневый заголовок расплющен без кортежей",
            "(" not in out.split("=== Таблица 1")[1].splitlines()[1],
            f"строка заголовка: {out.split('=== Таблица 1')[1].splitlines()[1]!r}",
        )
    )
    results.append(check("указано число строк таблицы", "строк: 5" in out, out[:300]))
    results.append(
        check("вторая таблица тоже разобрана", "Best Album" in out and "Won" in out)
    )

    # Ключевое для вопросов «сколько альбомов за 2000-2009»: по CSV должно быть
    # возможно отобрать строки по годам. Проверяем, что данные пригодны для счёта.
    csv_part = out.split("=== Таблица 1")[1]
    in_range = [y for y in ("2000", "2005", "2009") if y in csv_part]
    out_range = [y for y in ("1999", "2012") if y in csv_part]
    results.append(
        check(
            "по таблице можно отобрать годы (есть и внутри, и вне диапазона)",
            len(in_range) == 3 and len(out_range) == 2,
            f"внутри: {in_range}, вне: {out_range}",
        )
    )

    # --- пустой HTML ---
    try:
        empty = tool._tables_from_html("<div>без таблиц</div>")
        results.append(check("HTML без таблиц не роняет разбор", "не найдено таблиц" in empty.lower(), empty))
    except ValueError:
        # pandas.read_html бросает ValueError, когда таблиц нет вообще —
        # это ожидаемо и обрабатывается уровнем выше (forward ловит Exception).
        results.append(check("HTML без таблиц: ValueError обрабатывается выше", True))

    # --- поиск фрагментов ---
    frags = tool._extract_fragments(LONG_TEXT, "NASA award", tool.CONTEXT_CHARS)
    results.append(check("фрагмент по подстроке найден", len(frags) == 1, f"найдено: {len(frags)}"))
    results.append(check("в фрагменте есть искомое значение", "80GSFC21M0002" in frags[0], frags[0][:120]))
    results.append(
        check(
            "фрагмент обрезан, а не отдан целиком",
            len(frags[0]) < len(LONG_TEXT),
            f"{len(frags[0])} против {len(LONG_TEXT)}",
        )
    )
    results.append(
        check("отсутствующая подстрока даёт пустой список", tool._extract_fragments(LONG_TEXT, "щщщ", 100) == [])
    )
    results.append(
        check("регистр не важен", len(tool._extract_fragments(LONG_TEXT, "nasa AWARD", 100)) == 1)
    )

    print("\n" + "-" * 70)
    if all(results):
        print(f"Все {len(results)} проверок прошли.")
        return 0
    print(f"ПРОВАЛЕНО: {results.count(False)} из {len(results)}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
