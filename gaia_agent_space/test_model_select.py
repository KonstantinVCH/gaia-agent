"""
Тесты выбора рабочей модели — без сети и без токена.

Проверяется логика перебора: уважается ли MODEL_ID, действительно ли перебор
идёт дальше при недоступной модели, не проверяется ли одна модель дважды и
понятна ли ошибка, когда не работает ничего. Обращения к API подменяются
заглушкой: смысл теста — именно порядок и ветвления, а не живой инференс.

Запуск:  python test_model_select.py
"""

import os
import sys

import agent as agent_module


class FakeModel:
    """Заглушка InferenceClientModel: отвечает только на разрешённые имена."""

    working: set[str] = set()
    attempts: list[str] = []

    def __init__(self, model_id: str, **kwargs):
        self.model_id = model_id
        FakeModel.attempts.append(model_id)
        if model_id not in FakeModel.working:
            raise RuntimeError(f"model {model_id} is not served")

    def __call__(self, messages, **kwargs):
        return "pong"


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"[{'ok ' if condition else 'FAIL'}] {label}" + (f" — {detail}" if detail and not condition else ""))
    return condition


def run_case(working, candidates, model_id_env=None):
    FakeModel.working = set(working)
    FakeModel.attempts = []
    # Провайдер определяется по наличию ключа: без него select_working_model
    # сразу сообщает, что работать нечем, и до перебора дело не доходит.
    # Здесь проверяется именно перебор, поэтому ключ выставляем.
    os.environ["HF_TOKEN"] = "dummy-token-for-tests"
    os.environ.pop("GROQ_API_KEY", None)
    if model_id_env:
        os.environ["MODEL_ID"] = model_id_env
    else:
        os.environ.pop("MODEL_ID", None)
    try:
        _, chosen = agent_module.select_working_model(candidates, verbose=False)
        return chosen, list(FakeModel.attempts), None
    except RuntimeError as e:
        return None, list(FakeModel.attempts), str(e)


def main() -> int:
    # Подменяем и класс, и build_model: select_working_model создаёт модель
    # через build_model, а тот выбирает класс по провайдеру.
    agent_module.InferenceClientModel = FakeModel
    agent_module.build_model = lambda model_id, provider=None: FakeModel(model_id)
    results = []

    # Первая же модель доступна — до остальных дело не доходит.
    chosen, attempts, _ = run_case({"A"}, ["A", "B", "C"])
    results.append(check("берётся первая доступная", chosen == "A", f"выбрано {chosen}"))
    results.append(check("лишние варианты не проверяются", attempts == ["A"], f"попытки: {attempts}"))

    # Первая недоступна — перебор идёт дальше. Это главный смысл всей затеи.
    chosen, attempts, _ = run_case({"C"}, ["A", "B", "C"])
    results.append(check("перебор доходит до рабочей модели", chosen == "C", f"выбрано {chosen}"))
    results.append(check("проверены все до неё", attempts == ["A", "B", "C"], f"попытки: {attempts}"))

    # MODEL_ID имеет приоритет над встроенным списком.
    chosen, attempts, _ = run_case({"A", "Z"}, ["A", "B"], model_id_env="Z")
    results.append(check("MODEL_ID проверяется первым", chosen == "Z", f"выбрано {chosen}"))
    results.append(check("список не тронут, раз MODEL_ID сработал", attempts == ["Z"], f"попытки: {attempts}"))

    # MODEL_ID задан, но нерабочий — не должен блокировать остальные варианты.
    chosen, attempts, _ = run_case({"B"}, ["A", "B"], model_id_env="Z")
    results.append(check("нерабочий MODEL_ID не блокирует перебор", chosen == "B", f"выбрано {chosen}"))
    results.append(check("MODEL_ID шёл первым", attempts[0] == "Z", f"попытки: {attempts}"))

    # Дедупликация: MODEL_ID совпал с элементом списка — проверяем один раз.
    chosen, attempts, _ = run_case({"B"}, ["A", "B"], model_id_env="B")
    results.append(check("дубликат не проверяется дважды", attempts.count("B") == 1, f"попытки: {attempts}"))

    # None в списке (если smolagents не отдал свой дефолт) не должен ломать перебор.
    chosen, attempts, _ = run_case({"B"}, [None, "B"])
    results.append(check("None в списке пропускается", chosen == "B", f"выбрано {chosen}, попытки {attempts}"))

    # Не работает ничего — понятная ошибка с перечислением причин.
    chosen, attempts, err = run_case(set(), ["A", "B"])
    results.append(check("при полном отказе бросается ошибка", chosen is None and err is not None))
    results.append(check("в ошибке перечислены все кандидаты", err and "A" in err and "B" in err))
    results.append(check("в ошибке есть подсказка про MODEL_ID", err and "MODEL_ID" in err))
    results.append(check("в ошибке есть ссылка на список моделей", err and "inference/models" in err))

    print("\n" + "-" * 70)
    if all(results):
        print(f"Все {len(results)} проверок прошли.")
        return 0
    print(f"ПРОВАЛЕНО: {results.count(False)} из {len(results)}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
