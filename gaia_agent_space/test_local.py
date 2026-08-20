"""
Прогон агента на N вопросах из GAIA API без отправки результатов.
Использование:
  set HF_TOKEN=hf_...
  python test_local.py --count 3
"""

import argparse
import os
import sys

DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=3, help="Сколько вопросов прогнать")
    args = parser.parse_args()

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ Ошибка: переменная HF_TOKEN не задана.")
        print("   Выполните: set HF_TOKEN=hf_...")
        sys.exit(1)

    import requests
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from agent import GaiaAgent, DEFAULT_MODEL_ID

    api_url = DEFAULT_API_URL
    print(f"Загружаю вопросы с {api_url}/questions ...")
    r = requests.get(f"{api_url}/questions", timeout=15)
    r.raise_for_status()
    questions = r.json()[: args.count]
    print(f"Взял {len(questions)} вопрос(ов). Модель: {os.getenv('MODEL_ID', DEFAULT_MODEL_ID)}\n")

    agent = GaiaAgent(api_url=api_url)

    for i, item in enumerate(questions, 1):
        task_id = item.get("task_id")
        question = item.get("question", "")
        file_name = item.get("file_name") or ""
        print(f"── Вопрос {i}/{len(questions)} (task_id={task_id}) ──")
        print(f"Q: {question[:200]}")
        if file_name:
            print(f"   (Приложен файл: {file_name})")
        answer = agent(question, task_id=task_id, file_name=file_name)
        print(f"A: {answer}\n")


if __name__ == "__main__":
    main()
