"""
Локальный прогон агента по вопросам GAIA — БЕЗ публикации Space и БЕЗ отправки
ответов на сервер оценки.

Зачем: убедиться, что агент вообще работает (модель доступна, инструменты
вызываются, формат ответа корректный) до того, как выкладывать Space и тратить
единственную "настоящую" попытку. Здесь ничего не отправляется — только
локальный прогон и печать ответов.

Как запустить:

    export HF_TOKEN=ваш_токен_hugging_face      # в Windows: set HF_TOKEN=...
    pip install -r requirements.txt
    python test_local.py                        # один случайный вопрос
    python test_local.py --count 3              # три вопроса из общего набора
    python test_local.py --all                  # все 20 вопросов (долго!)

Токен нужен только вашему локальному процессу для обращения к Inference API —
он никуда, кроме Hugging Face, не уходит.
"""

import argparse
import os
import sys

import requests

DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"


def fetch_questions(api_url: str) -> list[dict]:
    """Забирает полный список учебных вопросов (публичный эндпоинт, без токена)."""
    resp = requests.get(f"{api_url}/questions", timeout=30)
    resp.raise_for_status()
    return resp.json()


def main() -> int:
    parser = argparse.ArgumentParser(description="Локальный прогон GAIA-агента без отправки ответов.")
    parser.add_argument("--count", type=int, default=1, help="Сколько вопросов прогнать (по умолчанию 1).")
    parser.add_argument("--all", action="store_true", help="Прогнать все вопросы набора.")
    parser.add_argument("--task-id", type=str, default=None, help="Прогнать один конкретный вопрос по task_id.")
    parser.add_argument("--api-url", type=str, default=DEFAULT_API_URL, help="Базовый URL API оценки.")
    args = parser.parse_args()

    if not os.getenv("HF_TOKEN"):
        print("ОШИБКА: не задана переменная окружения HF_TOKEN.", file=sys.stderr)
        print("Получите токен на https://huggingface.co/settings/tokens и задайте его:", file=sys.stderr)
        print("  Linux/macOS:  export HF_TOKEN=hf_...", file=sys.stderr)
        print("  Windows cmd:  set HF_TOKEN=hf_...", file=sys.stderr)
        return 1

    # Импортируем агента только после проверки токена, чтобы не ждать загрузку
    # зависимостей ради сообщения об ошибке.
    from agent import GaiaAgent

    try:
        questions = fetch_questions(args.api_url)
    except requests.exceptions.RequestException as e:
        print(f"ОШИБКА: не удалось получить вопросы с {args.api_url}: {e}", file=sys.stderr)
        return 1

    print(f"Всего вопросов в наборе: {len(questions)}")

    if args.task_id:
        selected = [q for q in questions if q.get("task_id") == args.task_id]
        if not selected:
            print(f"ОШИБКА: вопрос с task_id={args.task_id} не найден.", file=sys.stderr)
            return 1
    elif args.all:
        selected = questions
    else:
        selected = questions[: max(1, args.count)]

    agent = GaiaAgent(api_url=args.api_url)

    for i, item in enumerate(selected, start=1):
        task_id = item.get("task_id")
        question = item.get("question", "")
        file_name = item.get("file_name") or None

        print("\n" + "=" * 78)
        print(f"[{i}/{len(selected)}] task_id: {task_id}")
        if file_name:
            print(f"приложенный файл: {file_name}")
        print(f"вопрос: {question}")
        print("-" * 78)

        try:
            answer = agent(question=question, task_id=task_id, file_name=file_name)
        except Exception as e:  # агент не должен падать наружу, но на всякий случай
            answer = f"<исключение: {e}>"

        print(f"ОТВЕТ АГЕНТА: {answer}")

    print("\n" + "=" * 78)
    print("Прогон закончен. Ответы НИКУДА не отправлены — это локальная проверка.")
    print("Сверьте ответы глазами: если агент отвечает осмысленно и в нужном")
    print("формате (без пояснений и лишних слов), можно публиковать Space.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
