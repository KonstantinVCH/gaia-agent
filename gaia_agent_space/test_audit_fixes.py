"""
Регрессионные тесты на дефекты, найденные независимым аудитом кода.

Каждый тест здесь соответствует конкретной подтверждённой ошибке, которая
уже была в коде и проявилась бы на реальном прогоне. Файл отдельный
намеренно: так видно, что именно было сломано, и повторное появление любой
из этих ошибок сразу всплывёт.

Запуск:  python test_audit_fixes.py
"""

import os
import tempfile

from agent import normalize_answer
from tools import ReadTextFileTool, WikipediaTool


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"[{'ok ' if condition else 'FAIL'}] {label}" + (f" — {detail}" if detail and not condition else ""))
    return condition


def main() -> int:
    results = []

    # --- Дефект 1: gradio[oauth] в зависимостях ---
    # Без extra 'oauth' gr.LoginButton падает при старте, и Space не работает
    # вовсе. Проверяем состав requirements, а не поведение gradio: воспроизводить
    # запуск Space локально дорого, а нужное условие — именно наличие extra.
    reqs = open("requirements.txt", encoding="utf-8").read()
    results.append(check("в requirements указан gradio[oauth]", "gradio[oauth]" in reqs, reqs.splitlines()[0]))
    results.append(
        check(
            "нет строки с одиноким gradio",
            not any(line.strip() == "gradio" for line in reqs.splitlines()),
        )
    )

    # --- Дефект 2: нестроковые финальные значения ---
    # CodeAgent возвращает результат вычисления как есть; до исправления
    # список уходил на сервер питоновским repr.
    results.append(check("список приводится к перечислению через запятую",
                         normalize_answer(["broccoli", "celery"]) == "broccoli, celery",
                         normalize_answer(["broccoli", "celery"])))
    results.append(check("кортеж тоже", normalize_answer((1, 2, 3)) == "1, 2, 3"))
    results.append(check("вложенные числа форматируются, а не repr-ятся",
                         normalize_answer([1.0, 2.5]) == "1, 2.5",
                         normalize_answer([1.0, 2.5])))
    results.append(check("целое по значению float теряет .0", normalize_answer(3.0) == "3"))
    results.append(check("настоящая дробь сохраняется", normalize_answer(3.14) == "3.14"))
    results.append(check("bool не превращается в 1/0", normalize_answer(True) == "true"))

    # --- Дефект 3: разделители тысяч ломали список чисел ---
    # Это самый коварный из найденных: верный ответ превращался в мусор молча.
    results.append(check("список номеров страниц НЕ склеивается",
                         normalize_answer("132,133,134,197,245") == "132,133,134,197,245",
                         normalize_answer("132,133,134,197,245")))
    results.append(check("список трёхзначных не склеивается",
                         normalize_answer("100,200,300") == "100,200,300",
                         normalize_answer("100,200,300")))
    results.append(check("однозначный разделитель тысяч убирается", normalize_answer("1,234") == "1234"))
    results.append(check("большое число с разделителями убирается", normalize_answer("12,345,678") == "12345678"))
    results.append(check("пробел как разделитель тысяч убирается", normalize_answer("1 234") == "1234"))

    # --- Дефект 6: markdown-заборчик обнулял ответ ---
    results.append(check("значение в блоке кода извлекается", normalize_answer("```\n42\n```") == "42",
                         repr(normalize_answer("```\n42\n```"))))
    results.append(check("блок кода с языком тоже", normalize_answer("```python\n42\n```") == "42"))
    results.append(check("ответ в заборчике не становится пустым", normalize_answer("```\nParis\n```") != ""))

    # --- Дефект 8: срезка префикса откусывала часть ответа ---
    phrase = 'The word is "answer:" itself'
    results.append(check("слово answer внутри ответа не срезается",
                         normalize_answer(phrase) == phrase, normalize_answer(phrase)))
    results.append(check("настоящий префикс в начале всё ещё срезается",
                         normalize_answer("FINAL ANSWER: 42") == "42"))
    results.append(check("префикс с новой строки срезается",
                         normalize_answer("Reasoning here\nFINAL ANSWER: 42") == "42",
                         normalize_answer("Reasoning here\nFINAL ANSWER: 42")))

    # --- Дефект 4: чтение текстового файла (open в песочнице запрещён) ---
    tool = ReadTextFileTool()
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write("x = 2 + 3\nprint(x)\n")
        path = f.name
    content = tool.forward(path)
    results.append(check("текстовый файл читается", "x = 2 + 3" in content, content[:80]))
    results.append(check("отсутствующий файл даёт понятный текст",
                         "не найден" in tool.forward("/tmp/нет-такого-файла-12345.py").lower()))
    results.append(check("папка вместо файла обрабатывается",
                         "папка" in tool.forward(tempfile.gettempdir()).lower()))
    results.append(check("ограничение длины работает",
                         "обрезано" in tool.forward(path, max_chars=5)))
    os.unlink(path)

    # Битая кодировка не должна терять весь файл.
    with tempfile.NamedTemporaryFile("wb", suffix=".txt", delete=False) as f:
        f.write(b"good text \xff\xfe more text")
        bad_path = f.name
    out = tool.forward(bad_path)
    results.append(check("битые байты не роняют чтение", "good text" in out and "more text" in out, out[:80]))
    os.unlink(bad_path)

    # --- Дефект 7: опечатка в mode молча давала прозу вместо таблиц ---
    wiki = WikipediaTool()
    seen = []

    class FakeResp:
        status_code = 200

        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._payload

    def fake_get(url, params=None, headers=None, timeout=None):
        action = params.get("action")
        seen.append(action)
        if action == "query" and params.get("list") == "search":
            return FakeResp({"query": {"search": [{"title": "Test"}]}})
        if action == "parse":
            return FakeResp({"parse": {"text": {"*": "<table><tr><th>A</th></tr><tr><td>1</td></tr></table>"}}})
        return FakeResp({"query": {"pages": {"1": {"extract": "Plain prose, no tables."}}}})

    import tools as tools_module

    original_get = tools_module.requests.get
    tools_module.requests.get = fake_get
    try:
        for mode in ("tables", "table", "TABLES", " Tables "):
            seen.clear()
            out = wiki.forward("Test", mode=mode)
            results.append(
                check(
                    f"режим {mode!r} трактуется как таблицы",
                    "parse" in seen and "Найдено таблиц" in out,
                    f"запросы: {seen}, ответ: {out[:60]}",
                )
            )
        seen.clear()
        out = wiki.forward("Test", mode="text")
        results.append(check("режим 'text' по-прежнему даёт текст", "parse" not in seen and "prose" in out,
                             f"запросы: {seen}"))
    finally:
        tools_module.requests.get = original_get

    # --- язык, обращённый к модели ---
    # Описания инструментов и inputs — единственный текст, по которому модель
    # решает, что вызвать. Вопросы GAIA английские; русские описания ухудшали
    # выбор инструмента, из-за чего модель уходила в повторные web_search и
    # исчерпывала лимит шагов на реальном прогоне.
    import re as _re
    import tools as _tools

    tool_classes = [
        _tools.DownloadGaiaFileTool, _tools.TranscribeAudioTool,
        _tools.YoutubeTranscriptTool, _tools.DescribeImageTool,
        _tools.ReadPdfTool, _tools.WikipediaTool, _tools.ReadTextFileTool,
    ]
    for cls in tool_classes:
        results.append(
            check(
                f"описание {cls.name} без кириллицы",
                not _re.search("[а-яА-Я]", cls.description),
                cls.description[:70],
            )
        )
        cyrillic_inputs = [
            key for key, spec in cls.inputs.items()
            if _re.search("[а-яА-Я]", spec.get("description", ""))
        ]
        results.append(
            check(f"параметры {cls.name} без кириллицы", not cyrillic_inputs, str(cyrillic_inputs))
        )

    import agent as _agent

    for label, text in (
        ("ANSWER_INSTRUCTIONS", _agent.ANSWER_INSTRUCTIONS),
        ("EFFICIENCY_INSTRUCTIONS", _agent.EFFICIENCY_INSTRUCTIONS),
    ):
        results.append(check(f"{label} без кириллицы", not _re.search("[а-яА-Я]", text)))

    # Инструкция против циклов должна прямо запрещать повтор поиска — именно это
    # поведение наблюдалось на прогоне.
    results.append(
        check("в инструкции есть запрет повторять поиск",
              "NEVER repeat a search" in _agent.EFFICIENCY_INSTRUCTIONS)
    )
    results.append(
        check("в инструкции есть указание на mode='tables'",
              "mode='tables'" in _agent.EFFICIENCY_INSTRUCTIONS)
    )
    results.append(
        check("wikipedia описан как предпочтительный перед web_search",
              "PREFER THIS over" in _tools.WikipediaTool.description)
    )

    print("\n" + "-" * 70)
    if all(results):
        print(f"Все {len(results)} проверок прошли.")
        return 0
    print(f"ПРОВАЛЕНО: {results.count(False)} из {len(results)}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
