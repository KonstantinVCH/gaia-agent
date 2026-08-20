"""
Тесты для normalize_answer из agent.py.

Запуск (зависимости агента не нужны, HF_TOKEN не нужен):

    python test_normalize.py

Смысл: сервер оценки GAIA сверяет ответы дословно, поэтому нормализатор —
то место, где «правильный по смыслу» ответ легко превратить в неверный.
Здесь проверяется и то, что он чистит нужное, и то, что он НЕ портит
ответы, которые трогать нельзя.
"""

import importlib.util
import pathlib
import sys

# Импортируем agent.py напрямую, не подтягивая smolagents: сам модуль
# импортирует тяжёлые зависимости на верхнем уровне, а для теста чистой
# функции они не нужны. Поэтому читаем только определение функции.
_SOURCE = (pathlib.Path(__file__).parent / "agent.py").read_text(encoding="utf-8")
# Берём с _format_scalar: normalize_answer на него опирается, и вырезать
# только её означает получить NameError вместо теста.
_START = _SOURCE.index("def _format_scalar")
_END = _SOURCE.index("class GaiaAgent")
_NAMESPACE: dict = {}
exec("import re\nimport numbers\n" + _SOURCE[_START:_END], _NAMESPACE)
normalize_answer = _NAMESPACE["normalize_answer"]


# (вход, ожидаемый выход, зачем этот случай)
CASES = [
    # --- то, что нужно почистить ---
    ("42", "42", "уже чистый ответ не меняется"),
    ("  42  ", "42", "обрезка пробелов"),
    ("FINAL ANSWER: 42", "42", "срез служебного префикса"),
    ("final answer: 42", "42", "префикс в нижнем регистре"),
    ("Ответ: 42", "42", "русский префикс"),
    ("Answer - 42", "42", "префикс через дефис"),
    ("The answer: 42.", "42", "префикс + точка в конце"),
    ("**42**", "42", "markdown-выделение"),
    ('"Paris"', "Paris", "обрамляющие кавычки"),
    ("«Paris»", "Paris", "русские кавычки"),
    ("`42`", "42", "обратные кавычки"),
    ("Paris.", "Paris", "точка в конце"),
    ("1,234", "1234", "разделитель тысяч запятой"),
    ("1 234 567", "1234567", "разделитель тысяч пробелом"),
    ("Рассуждения тут\nFINAL ANSWER: 7", "7", "берём хвост после последнего префикса"),
    ("строка один\nстрока два\n42", "42", "из многострочного берём последнюю строку"),

    # --- то, что портить НЕЛЬЗЯ ---
    ("3.14", "3.14", "десятичная точка не срезается"),
    ("3.14.", "3.14", "точка в конце снимается, дробь остаётся целой"),
    ("42.", "42", "точка в конце числа снимается"),
    ("U.S.A.", "U.S.A.", "точки в аббревиатуре значимы"),
    ("Paris, France", "Paris, France", "запятая в списке — не разделитель тысяч"),
    ("apple, banana, cherry", "apple, banana, cherry", "список через запятую сохраняется"),
    ("St. Petersburg", "St. Petersburg", "точка внутри строки не трогается"),
    ("Rome", "Rome", "имя собственное — регистр сохраняется"),
    ("CIA", "CIA", "аббревиатура в верхнем регистре сохраняется"),
    ("", "", "пустая строка"),
    (None, "", "None не роняет функцию"),
    (42, "42", "не-строка приводится к строке"),
]


def main() -> int:
    failures = []
    for raw, expected, why in CASES:
        got = normalize_answer(raw)
        status = "ok " if got == expected else "FAIL"
        if got != expected:
            failures.append((raw, expected, got, why))
        print(f"[{status}] {raw!r:45} -> {got!r:22} ({why})")

    print("\n" + "-" * 70)
    if failures:
        print(f"ПРОВАЛЕНО: {len(failures)} из {len(CASES)}")
        for raw, expected, got, why in failures:
            print(f"  {raw!r}: ожидалось {expected!r}, получено {got!r}  [{why}]")
        return 1

    print(f"Все {len(CASES)} проверок прошли.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
