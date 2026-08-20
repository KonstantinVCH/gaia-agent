"""
Запускает агента на всех вопросах GAIA локально и отправляет ответы
напрямую на сервер оценки — без Hugging Face Space.

Использование:
  set GOOGLE_API_KEY=...
  set HF_USERNAME=KonstantinVCH
  python submit_local.py

Вместо Space-URL в качестве agent_code используется GitHub-репозиторий.
"""

import os
import sys
import json
import requests

DEFAULT_API_URL  = "https://agents-course-unit4-scoring.hf.space"
AGENT_CODE_URL   = "https://github.com/KonstantinVCH/gaia-agent"


def main():
    username = os.getenv("HF_USERNAME", "KonstantinVCH")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from agent import GaiaAgent

    api_url = DEFAULT_API_URL

    # 1. Загружаем вопросы
    print(f"Загружаю вопросы с {api_url}/questions ...")
    r = requests.get(f"{api_url}/questions", timeout=15)
    r.raise_for_status()
    questions = r.json()
    print(f"Всего вопросов: {len(questions)}\n")

    # 2. Инициализируем агента
    agent = GaiaAgent(api_url=api_url)

    # 3. Прогоняем все вопросы
    answers_payload = []
    results_log = []
    for i, item in enumerate(questions, 1):
        task_id    = item.get("task_id")
        question   = item.get("question", "")
        file_name  = item.get("file_name") or ""
        print(f"[{i:02d}/{len(questions)}] {question[:80]}")
        answer = agent(question, task_id=task_id, file_name=file_name)
        print(f"       → {answer}\n")
        answers_payload.append({"task_id": task_id, "submitted_answer": answer})
        results_log.append({"Task ID": task_id, "Question": question, "Answer": answer})

    # 4. Отправляем
    submission = {
        "username": username,
        "agent_code": AGENT_CODE_URL,
        "answers": answers_payload,
    }
    print(f"Отправляю {len(answers_payload)} ответов от имени {username} ...")
    resp = requests.post(f"{api_url}/submit", json=submission, timeout=60)
    resp.raise_for_status()
    result = resp.json()

    print("\n" + "="*50)
    print(f"Результат: {result.get('score', '?')}%  "
          f"({result.get('correct_count', '?')}/{result.get('total_attempted', '?')} верных)")
    print(f"Сообщение: {result.get('message', '')}")
    print("="*50)

    # Сохраняем таблицу
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results_log, f, ensure_ascii=False, indent=2)
    print(f"\nТаблица ответов → results.json")
    return result.get("score", 0)


if __name__ == "__main__":
    main()
