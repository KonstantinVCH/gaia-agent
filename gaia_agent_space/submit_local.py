"""
Прогон агента и отправка ответов БЕЗ публикации Hugging Face Space.

Зачем это нужно. Официальная инструкция курса предполагает публичный Space на
Gradio, но Hugging Face закрыл free-аккаунтам создание CPU-Basic Gradio Spaces
(остаётся ZeroGPU, а перевод на CPU Basic требует PRO). То есть путь из
инструкции для бесплатного аккаунта попросту недоступен.

Обход опирается на факт, а не на трюк: эндпоинт `/submit` сервера оценки не
требует авторизации — он принимает обычный POST с полями `username`,
`agent_code` и списком ответов. OAuth в Space нужен был лишь для того, чтобы
надёжно узнать имя пользователя. Здесь имя указывается явным аргументом.

Что важно понимать про честность этого пути. Требование курса про публичный
Space — про то, чтобы код можно было проверить. Поэтому `--agent-code` должен
указывать на публично доступный код (например, ваш репозиторий на GitHub).
Отвечает по-прежнему агент, ответы не подставляются вручную. Но это отклонение
от буквы инструкции, и в лидерборде ссылка будет вести на GitHub, а не на
Space — решать вам.

Скрипт разделён на два шага намеренно: прогон 20 вопросов занимает минуты, и
терять его из-за сетевой ошибки при отправке нельзя. Поэтому ответы сначала
сохраняются в файл, а отправка — отдельная операция, которую можно повторить.

Примеры:
    # 1. Прогон с сохранением в файл (ничего не отправляется)
    python submit_local.py run --all

    # 2. Посмотреть, что уйдёт на сервер
    python submit_local.py submit --username ВАШ_НИК --dry-run

    # 3. Отправить
    python submit_local.py submit --username ВАШ_НИК \
        --agent-code https://github.com/ВАШ_НИК/gaia-agent

    # Всё сразу (прогон + отправка с подтверждением)
    python submit_local.py all --username ВАШ_НИК --agent-code https://...
"""

import argparse
import json
import os
import sys

import requests

DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
ANSWERS_FILE = "answers.json"


def fetch_questions(api_url: str) -> list:
    response = requests.get(f"{api_url}/questions", timeout=30)
    response.raise_for_status()
    questions = response.json()
    if not isinstance(questions, list) or not questions:
        raise RuntimeError(f"Сервер вернул неожиданный список вопросов: {questions!r}")
    return questions


def load_answers(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_answers(path: str, answers: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)


def do_run(args) -> int:
    """Прогоняет агента по вопросам и складывает ответы в файл."""
    from agent import GaiaAgent, active_provider

    provider = active_provider()
    if provider == "none":
        print(
            "ОШИБКА: не задан ни один ключ. Нужен GROQ_API_KEY (бесплатно, без карты:\n"
            "https://console.groq.com/keys) либо HF_TOKEN.",
            file=sys.stderr,
        )
        return 1

    questions = fetch_questions(args.api_url)
    print(f"Вопросов получено: {len(questions)} | провайдер: {provider}")

    # Уже отвеченные вопросы не прогоняем заново: если прошлый запуск прервался
    # на середине, повторять уже сделанную работу незачем.
    existing = {a["task_id"]: a for a in load_answers(args.answers)}
    if existing:
        print(f"В {args.answers} уже есть ответов: {len(existing)} — они будут сохранены.")

    todo = [q for q in questions if q.get("task_id") not in existing]
    if args.count:
        todo = todo[: args.count]
    if not todo:
        print("Все вопросы уже отвечены. Переходите к шагу submit.")
        return 0

    print(f"К прогону: {len(todo)}\n")
    agent = GaiaAgent(api_url=args.api_url)

    for index, item in enumerate(todo, start=1):
        task_id = item.get("task_id")
        question = item.get("question") or ""
        file_name = item.get("file_name") or None

        print(f"[{index}/{len(todo)}] {task_id}")
        print(f"    {question[:120]}{'...' if len(question) > 120 else ''}")
        try:
            answer = agent(question, task_id=task_id, file_name=file_name)
        except Exception as e:
            # Один упавший вопрос не должен обрушивать весь прогон: остальные
            # ответы всё равно нужны, порог считается по сумме.
            answer = f"AGENT ERROR: {type(e).__name__}"
            print(f"    ошибка: {e}")
        print(f"    ответ: {answer!r}\n")

        existing[task_id] = {"task_id": task_id, "submitted_answer": answer}
        # Сохраняем после КАЖДОГО вопроса, а не в конце: прогон длинный, и
        # обрыв на середине не должен обнулять уже полученные ответы.
        save_answers(args.answers, list(existing.values()))

    print(f"Готово. Ответов в {args.answers}: {len(existing)}")
    print(f"Дальше: python submit_local.py submit --username ВАШ_НИК --agent-code ССЫЛКА")
    return 0


def do_submit(args) -> int:
    """Отправляет сохранённые ответы на сервер оценки."""
    answers = load_answers(args.answers)
    if not answers:
        print(f"ОШИБКА: в {args.answers} нет ответов. Сначала: python submit_local.py run --all",
              file=sys.stderr)
        return 1

    if not args.username:
        print("ОШИБКА: укажите --username (ваш ник на Hugging Face).", file=sys.stderr)
        return 1

    if not args.agent_code:
        print(
            "ОШИБКА: укажите --agent-code — публичную ссылку на код агента.\n"
            "Требование курса: код должен быть доступен для проверки. Подойдёт\n"
            "ваш публичный репозиторий на GitHub.",
            file=sys.stderr,
        )
        return 1

    payload = {
        "username": args.username.strip(),
        "agent_code": args.agent_code.strip(),
        "answers": answers,
    }

    errors = [a for a in answers if str(a["submitted_answer"]).startswith(("AGENT ERROR", "UNKNOWN"))]

    print(f"Пользователь:  {payload['username']}")
    print(f"Ссылка на код: {payload['agent_code']}")
    print(f"Ответов:       {len(answers)}" + (f" (из них неудачных: {len(errors)})" if errors else ""))
    print(f"Адрес:         {args.api_url}/submit")

    if args.dry_run:
        print("\n--dry-run: ничего не отправлено. Содержимое:")
        print(json.dumps(payload, ensure_ascii=False, indent=2)[:2000])
        return 0

    # Отправка публикует результат в лидерборд под вашим именем, поэтому
    # требуется явное подтверждение. Отменить отправку нельзя.
    if not args.yes:
        print("\nЭто опубликует результат под вашим именем. Отменить нельзя.")
        reply = input("Отправить? (введите да): ").strip().lower()
        if reply not in ("да", "yes", "y"):
            print("Отменено.")
            return 1

    try:
        response = requests.post(f"{args.api_url}/submit", json=payload, timeout=120)
    except requests.exceptions.RequestException as e:
        print(f"ОШИБКА сети при отправке: {e}", file=sys.stderr)
        print(f"Ответы сохранены в {args.answers} — повторите submit, прогон не нужен.", file=sys.stderr)
        return 1

    if response.status_code >= 400:
        print(f"\nСервер отказал: HTTP {response.status_code}", file=sys.stderr)
        print(response.text[:1500], file=sys.stderr)
        print(
            f"\nОтветы целы в {args.answers}. Если причина в поле agent_code — "
            f"попробуйте другую публичную ссылку и повторите submit.",
            file=sys.stderr,
        )
        return 1

    try:
        result = response.json()
    except ValueError:
        print(f"Сервер вернул не-JSON: {response.text[:500]}")
        return 1

    print("\n=== Результат ===")
    print(f"Пользователь:  {result.get('username')}")
    print(f"Счёт:          {result.get('score', '?')}%")
    print(f"Правильных:    {result.get('correct_count', '?')} из {result.get('total_attempted', '?')}")
    print(f"Сообщение:     {result.get('message', '')}")

    score = result.get("score")
    try:
        passed = float(score) >= 30
    except (TypeError, ValueError):
        passed = False

    if passed:
        print("\nПорог 30% пройден — можно забирать сертификат на странице курса")
        print("(Unit 4 -> Get Your Certificate).")
    else:
        print("\nПорог 30% не пройден. Ответы остались в", args.answers)
        print("Что обычно помогает: MODEL_ID с более сильной моделью и повторный прогон")
        print("(удалите answers.json, чтобы прогнать заново).")
    return 0


def do_all(args) -> int:
    code = do_run(args)
    if code != 0:
        return code
    print()
    return do_submit(args)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Прогон агента и отправка ответов GAIA без публикации Space.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", choices=["run", "submit", "all"],
                        help="run — прогнать и сохранить; submit — отправить сохранённое; all — всё сразу.")
    parser.add_argument("--username", help="Ваш ник на Hugging Face (нужен для submit).")
    parser.add_argument("--agent-code", help="Публичная ссылка на код агента (например, репозиторий GitHub).")
    parser.add_argument("--answers", default=ANSWERS_FILE, help=f"Файл с ответами (по умолчанию {ANSWERS_FILE}).")
    parser.add_argument("--count", type=int, help="Прогнать только N вопросов (для пробы).")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Адрес сервера оценки.")
    parser.add_argument("--dry-run", action="store_true", help="Показать, что уйдёт, и не отправлять.")
    parser.add_argument("--yes", action="store_true", help="Не спрашивать подтверждение перед отправкой.")
    parser.add_argument("--all", action="store_true", help="Прогнать все вопросы (равнозначно отсутствию --count).")
    args = parser.parse_args()

    if args.all:
        args.count = None

    handlers = {"run": do_run, "submit": do_submit, "all": do_all}
    try:
        return handlers[args.command](args)
    except KeyboardInterrupt:
        print("\nПрервано. Уже полученные ответы сохранены.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
