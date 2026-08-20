"""
Тест, который реально поднимает app.py в окружении, похожем на Hugging Face Space.

Зачем отдельный тест. Дважды за разработку код ломался именно так: собирается
и импортируется локально, но на Space падает при старте — сначала из-за
переименованного пакета `ddgs`, потом из-за отсутствия `gradio[oauth]`.
Оба раза остальные тесты были зелёными, потому что подменяли заглушками как раз
те места, которые ломаются: кнопку входа и создание агента. Тест, который
ничего не подменяет и просто пробует запустить приложение так, как это делает
Space, ловит весь этот класс ошибок целиком.

Что проверяется:
  - app.py импортируется при выставленных переменных окружения Space;
  - интерфейс собирается, и на нём действительно включён вход через HF;
  - зависимости для OAuth реально установлены (без них падает gr.LoginButton);
  - каждый пакет из requirements.txt импортируется.

Сеть и HF_TOKEN не нужны: агент создаётся уже по нажатию кнопки, а не при импорте.

Запуск:  python test_space_startup.py
"""

import os

# ВАЖНО: окружение Space выставляется до любых импортов gradio.
# gradio/oauth.py читает OAUTH_CLIENT_ID на уровне модуля, то есть значение
# фиксируется в момент импорта. Если сначала импортировать gradio (пусть даже
# косвенно, проверяя зависимости), а окружение поправить потом — приложение
# упадёт с «OAUTH_CLIENT_ID is not set», хотя переменная выставлена.
# Этот тест на такую ошибку уже наступил, поэтому порядок здесь существенный.
SPACE_ENV = {
    "SYSTEM": "spaces",
    "SPACE_ID": "test-user/gaia-agent",
    "SPACE_HOST": "test-user-gaia-agent.hf.space",
    "OAUTH_CLIENT_ID": "dummy-client-id",
    "OAUTH_CLIENT_SECRET": "dummy-client-secret",
    "OAUTH_SCOPES": "openid profile",
    "OPENID_PROVIDER_URL": "https://huggingface.co",
}
os.environ.update(SPACE_ENV)

import importlib
import re
import sys


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"[{'ok ' if condition else 'FAIL'}] {label}" + (f" — {detail}" if detail and not condition else ""))
    return condition


REQUIREMENTS_TO_MODULE = {
    "gradio[oauth]": "gradio",
    "requests": "requests",
    "pandas": "pandas",
    "openpyxl": "openpyxl",
    "smolagents": "smolagents",
    "ddgs": "ddgs",
    "markdownify": "markdownify",
    "huggingface_hub": "huggingface_hub",
    "youtube-transcript-api": "youtube_transcript_api",
    "pypdf": "pypdf",
    "lxml": "lxml",
    # openai нужен для OpenAIServerModel — через него идёт путь на Groq.
    "openai": "openai",
}


def main() -> int:
    results = []

    # --- зависимости OAuth: именно их отсутствие ломало Space ---
    for module in ("authlib", "itsdangerous"):
        try:
            importlib.import_module(module)
            ok = True
        except ImportError:
            ok = False
        results.append(
            check(
                f"установлен {module} (приходит с gradio[oauth])",
                ok,
                "поставьте: pip install 'gradio[oauth]'",
            )
        )

    # --- каждый пакет из requirements.txt импортируется ---
    lines = [l.strip() for l in open("requirements.txt", encoding="utf-8") if l.strip()]

    # Явная и понятная проверка именно того, на чём код уже ломался: бареный
    # `gradio` вместо `gradio[oauth]`. Без неё ошибка всплывала бы невнятным
    # сообщением «строка неизвестна тесту».
    results.append(
        check(
            "в requirements указан gradio[oauth], а не просто gradio",
            "gradio[oauth]" in lines and "gradio" not in lines,
            "gr.LoginButton() требует extra 'oauth' (пакет authlib) — без него Space падает при старте",
        )
    )

    for line in lines:
        module = REQUIREMENTS_TO_MODULE.get(line)
        if not module:
            results.append(check(f"строка requirements '{line}' известна тесту", False,
                                 "добавьте её в REQUIREMENTS_TO_MODULE"))
            continue
        try:
            importlib.import_module(module)
            ok = True
            detail = ""
        except ImportError as e:
            ok, detail = False, str(e)[:100]
        results.append(check(f"импортируется {line} -> {module}", ok, detail))

    # --- собственно запуск приложения в окружении Space ---
    sys.modules.pop("app", None)
    app_module = None
    try:
        app_module = importlib.import_module("app")
        results.append(check("app.py импортируется в окружении Space", True))
    except Exception as e:
        results.append(check("app.py импортируется в окружении Space", False, f"{type(e).__name__}: {e}"))

    if app_module is not None:
        demo = getattr(app_module, "demo", None)
        results.append(check("интерфейс demo создан", demo is not None))
        if demo is not None:
            # expects_oauth выставляется самим gradio при наличии LoginButton.
            # Если бы кнопки не было, отправка ответов шла бы без имени
            # пользователя, и результат не зачёлся бы.
            results.append(
                check(
                    "на интерфейсе включён вход через Hugging Face",
                    getattr(demo, "expects_oauth", False),
                    "gr.LoginButton() отсутствует или не сработал",
                )
            )
        results.append(
            check(
                "функция прогона и отправки на месте",
                callable(getattr(app_module, "run_and_submit_all", None)),
            )
        )
        results.append(
            check(
                "адрес сервера оценки не подменён",
                getattr(app_module, "DEFAULT_API_URL", "") == "https://agents-course-unit4-scoring.hf.space",
                getattr(app_module, "DEFAULT_API_URL", "<нет>"),
            )
        )

    # --- README со шапкой для Space ---
    readme = open("README.md", encoding="utf-8").read()
    results.append(check("README начинается с YAML-шапки", readme.startswith("---")))
    for field in ("sdk: gradio", "app_file: app.py", "hf_oauth: true"):
        results.append(check(f"в шапке есть {field}", field in readme))
    version = re.search(r"sdk_version:\s*([\d.]+)", readme)
    results.append(check("в шапке указана версия sdk", version is not None,
                         "без sdk_version Space возьмёт версию наугад"))

    print("\n" + "-" * 70)
    if all(results):
        print(f"Все {len(results)} проверок прошли.")
        return 0
    print(f"ПРОВАЛЕНО: {results.count(False)} из {len(results)}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
