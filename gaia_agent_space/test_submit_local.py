"""
Тесты локальной отправки — без сети и без ключей.

Проверяется то, что дорого сломать: докачка ответов после обрыва (прогон 20
вопросов длинный, и терять сделанное нельзя), сохранение после каждого вопроса,
сборка payload и требование подтверждения перед публичной отправкой.

Запуск:  python test_submit_local.py
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import submit_local


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"[{'ok ' if condition else 'FAIL'}] {label}" + (f" — {detail}" if detail and not condition else ""))
    return condition


class Args:
    def __init__(self, **kw):
        self.api_url = submit_local.DEFAULT_API_URL
        self.answers = None
        self.count = None
        self.username = None
        self.agent_code = None
        self.dry_run = False
        self.yes = False
        self.__dict__.update(kw)


QUESTIONS = [
    {"task_id": "q1", "question": "Первый вопрос", "file_name": ""},
    {"task_id": "q2", "question": "Второй вопрос", "file_name": "f.mp3"},
    {"task_id": "q3", "question": "Третий вопрос", "file_name": ""},
]


def main() -> int:
    results = []
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "answers.json")

    # --- сохранение и чтение ---
    results.append(check("отсутствующий файл читается как пустой список", submit_local.load_answers(path) == []))
    submit_local.save_answers(path, [{"task_id": "x", "submitted_answer": "1"}])
    results.append(check("сохранённое читается обратно",
                         submit_local.load_answers(path) == [{"task_id": "x", "submitted_answer": "1"}]))

    # Русские ответы не должны превращаться в \uXXXX — файл читают глазами.
    submit_local.save_answers(path, [{"task_id": "y", "submitted_answer": "Санкт-Петербург"}])
    raw = open(path, encoding="utf-8").read()
    results.append(check("русский текст сохраняется читаемо", "Санкт-Петербург" in raw, raw[:80]))

    # --- прогон: докачка и сохранение после каждого вопроса ---
    submit_local.fetch_questions = lambda api_url: QUESTIONS

    calls = []
    snapshots = []
    real_save = submit_local.save_answers

    def spy_save(p, answers):
        snapshots.append(len(answers))
        real_save(p, answers)

    class FakeAgent:
        def __init__(self, api_url):
            pass

        def __call__(self, question, task_id=None, file_name=None):
            calls.append(task_id)
            return f"ответ-{task_id}"

    import agent as agent_module

    saved_agent, saved_provider = agent_module.GaiaAgent, agent_module.active_provider
    agent_module.GaiaAgent = FakeAgent
    agent_module.active_provider = lambda: "groq"
    submit_local.save_answers = spy_save
    try:
        fresh = os.path.join(tmp, "run.json")
        code = submit_local.do_run(Args(answers=fresh))
        results.append(check("прогон завершается успешно", code == 0))
        results.append(check("прогнаны все три вопроса", calls == ["q1", "q2", "q3"], str(calls)))
        results.append(
            check("сохранение после каждого вопроса, а не только в конце",
                  snapshots == [1, 2, 3], str(snapshots))
        )

        # Обрыв на середине: второй запуск должен добрать только недостающее.
        partial = os.path.join(tmp, "partial.json")
        real_save(partial, [{"task_id": "q1", "submitted_answer": "уже есть"}])
        calls.clear()
        submit_local.do_run(Args(answers=partial))
        results.append(check("после обрыва добираются только недостающие", calls == ["q2", "q3"], str(calls)))
        stored = {a["task_id"]: a["submitted_answer"] for a in submit_local.load_answers(partial)}
        results.append(check("прежний ответ не перезаписан", stored["q1"] == "уже есть", str(stored)))
        results.append(check("итого все три ответа на месте", len(stored) == 3, str(stored)))

        # --count ограничивает прогон
        limited = os.path.join(tmp, "limited.json")
        calls.clear()
        submit_local.do_run(Args(answers=limited, count=2))
        results.append(check("--count ограничивает число вопросов", calls == ["q1", "q2"], str(calls)))

        # Падение агента на одном вопросе не должно рушить остальные.
        class BoomAgent(FakeAgent):
            def __call__(self, question, task_id=None, file_name=None):
                if task_id == "q2":
                    raise RuntimeError("провайдер отвалился")
                return f"ответ-{task_id}"

        agent_module.GaiaAgent = BoomAgent
        boom = os.path.join(tmp, "boom.json")
        code = submit_local.do_run(Args(answers=boom))
        stored = {a["task_id"]: a["submitted_answer"] for a in submit_local.load_answers(boom)}
        results.append(check("падение на одном вопросе не рушит прогон", code == 0 and len(stored) == 3, str(stored)))
        results.append(check("упавший вопрос помечен как ошибка", stored["q2"].startswith("AGENT ERROR"), stored["q2"]))
        results.append(check("остальные ответы получены", stored["q1"] == "ответ-q1" and stored["q3"] == "ответ-q3"))
    finally:
        submit_local.save_answers = real_save
        agent_module.GaiaAgent, agent_module.active_provider = saved_agent, saved_provider

    # --- отправка: обязательные поля и dry-run ---
    ready = os.path.join(tmp, "ready.json")
    real_save(ready, [{"task_id": "q1", "submitted_answer": "4"}])

    results.append(check("submit без username отклоняется",
                         submit_local.do_submit(Args(answers=ready, agent_code="https://x")) == 1))
    results.append(check("submit без agent_code отклоняется",
                         submit_local.do_submit(Args(answers=ready, username="kv")) == 1))
    results.append(check("submit без файла ответов отклоняется",
                         submit_local.do_submit(Args(answers=os.path.join(tmp, "нет.json"),
                                                     username="kv", agent_code="https://x")) == 1))

    # dry-run ничего не отправляет: если бы отправлял, подмена post это поймала бы.
    posted = []
    real_post = submit_local.requests.post
    submit_local.requests.post = lambda *a, **k: posted.append(a) or (_ for _ in ()).throw(
        AssertionError("dry-run не должен отправлять")
    )
    try:
        code = submit_local.do_submit(
            Args(answers=ready, username="kv", agent_code="https://github.com/u/r", dry_run=True)
        )
        results.append(check("dry-run завершается успешно и молча", code == 0 and posted == []))
    finally:
        submit_local.requests.post = real_post

    print("\n" + "-" * 70)
    if all(results):
        print(f"Все {len(results)} проверок прошли.")
        return 0
    print(f"ПРОВАЛЕНО: {results.count(False)} из {len(results)}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
