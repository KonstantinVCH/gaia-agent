"""
Смоук-тест приложения: прогоняет run_and_submit_all целиком на заглушках —
без сети, без токена, без настоящей модели.

Проверяет ровно те вещи, которые молча ломаются и которые невозможно заметить,
глядя на код:
  1. UI вообще собирается на установленной версии Gradio;
  2. функция действительно генератор и выдаёт промежуточный прогресс;
  3. ИТОГОВЫЙ статус (со счётом) доходит до интерфейса — в генераторе
     `return значение` ничего не отдаёт, и результат прогона потерялся бы;
  4. без входа в аккаунт возвращается просьба залогиниться, а не падение;
  5. ошибка отправки не роняет приложение, а показывается пользователю.

Запуск:  python test_app_flow.py
"""

import sys
import types

import gradio as gr

# --- Заглушка кнопки входа.
# gr.LoginButton вне окружения Space требует, чтобы машина была залогинена в
# Hugging Face, и без этого падает ещё на сборке интерфейса. Это ограничение
# локальной отладки Gradio, а не дефект приложения: на самом Space вход
# работает штатно. Здесь проверяется логика прогона, поэтому кнопку подменяем
# обычной — иначе тест невозможно запустить без токена.
_REAL_LOGIN_BUTTON = gr.LoginButton
gr.LoginButton = lambda *a, **kw: gr.Button("Sign in with Hugging Face (mock)")


# --- Заглушка агента: подставляем модуль до импорта app, чтобы не тянуть
# --- настоящую модель и не требовать HF_TOKEN.
class _FakeAgent:
    def __init__(self, api_url, **kwargs):
        self.api_url = api_url

    def __call__(self, question, task_id=None, file_name=None):
        return f"answer-for-{task_id}"


fake_agent_module = types.ModuleType("agent")
fake_agent_module.GaiaAgent = _FakeAgent
sys.modules["agent"] = fake_agent_module

import app  # noqa: E402  (импорт после подмены — это осознанно)


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.text = str(payload)

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


QUESTIONS = [
    {"task_id": "t1", "question": "Сколько будет два плюс два?", "Level": "1", "file_name": ""},
    {"task_id": "t2", "question": "Столица Франции?", "Level": "1", "file_name": ""},
]

SUBMIT_RESULT = {
    "username": "tester",
    "score": 50.0,
    "correct_count": 1,
    "total_attempted": 2,
    "message": "Well done",
}


class _FakeProfile:
    username = "tester"


def install_fake_network(submit_raises: Exception | None = None):
    """Подменяем requests внутри модуля app."""

    def fake_get(url, timeout=None):
        assert "/questions" in url, f"неожиданный GET: {url}"
        return _FakeResponse(QUESTIONS)

    def fake_post(url, json=None, timeout=None):
        assert "/submit" in url, f"неожиданный POST: {url}"
        if submit_raises:
            raise submit_raises
        # проверяем, что payload собран правильно
        assert json["username"] == "tester", json
        assert len(json["answers"]) == len(QUESTIONS), json
        assert json["answers"][0]["submitted_answer"] == "answer-for-t1", json
        return _FakeResponse(SUBMIT_RESULT)

    app.requests.get = fake_get
    app.requests.post = fake_post


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"[{'ok ' if condition else 'FAIL'}] {label}" + (f" — {detail}" if detail and not condition else ""))
    return condition


def main() -> int:
    results = []

    # 1. UI собрался (это уже произошло при импорте app)
    results.append(check("интерфейс Gradio собирается", app.demo is not None))
    import gradio as gr

    print(f"       (проверено на gradio {gr.__version__})")

    # 2. функция — генератор
    import inspect

    results.append(
        check("run_and_submit_all — генератор", inspect.isgeneratorfunction(app.run_and_submit_all))
    )

    # 3. без входа — просьба залогиниться, без исключения
    out = list(app.run_and_submit_all(None))
    results.append(
        check(
            "без входа возвращается просьба залогиниться",
            len(out) == 1 and "Login" in out[0][0],
            f"получено: {out}",
        )
    )

    # 4. успешный прогон
    install_fake_network()
    steps = list(app.run_and_submit_all(_FakeProfile()))
    statuses = [s[0] for s in steps]

    results.append(check("промежуточный прогресс выдаётся", len(steps) > 3, f"шагов: {len(steps)}"))
    results.append(
        check(
            "виден прогресс по вопросам",
            any("Вопрос 1 из 2" in s for s in statuses),
            f"статусы: {statuses}",
        )
    )
    results.append(
        check(
            "ИТОГОВЫЙ статус со счётом доходит до интерфейса",
            "Submission Successful" in statuses[-1] and "50.0%" in statuses[-1],
            f"последний статус: {statuses[-1]!r}",
        )
    )
    last_table = steps[-1][1]
    results.append(
        check(
            "итоговая таблица заполнена",
            last_table is not None and len(last_table) == len(QUESTIONS),
            f"таблица: {last_table}",
        )
    )

    # 5. ошибка отправки не роняет приложение
    import requests as real_requests

    install_fake_network(submit_raises=real_requests.exceptions.Timeout())
    steps_fail = list(app.run_and_submit_all(_FakeProfile()))
    results.append(
        check(
            "таймаут отправки показывается пользователем, а не падает",
            "timed out" in steps_fail[-1][0],
            f"последний статус: {steps_fail[-1][0]!r}",
        )
    )

    print("\n" + "-" * 70)
    if all(results):
        print(f"Все {len(results)} проверок прошли.")
        return 0
    print(f"ПРОВАЛЕНО: {results.count(False)} из {len(results)}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
