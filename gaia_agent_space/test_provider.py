"""
Тесты выбора провайдера инференса — без сети и без реальных ключей.

Зачем. Изначально агент умел работать только через Hugging Face, и когда
месячный лимит HF исчерпался, он перестал отвечать вообще — при полностью
рабочем коде. Теперь провайдер выбирается по тому, какой ключ задан, и это
поведение нужно закрепить тестами: молчаливый откат не на тот провайдер
означает потраченный впустую прогон всех 20 вопросов.

Проверяется: определение провайдера по окружению, приоритет Groq над HF,
правильный класс и адрес модели, а для инструментов речи и зрения — что при
падении Groq происходит откат на HF, а не потеря вопроса.

Запуск:  python test_provider.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import agent as agent_module
import tools as tools_module

KEYS = ("GROQ_API_KEY", "HF_TOKEN", "MODEL_ID")


def check(label: str, condition: bool, detail: str = "") -> bool:
    print(f"[{'ok ' if condition else 'FAIL'}] {label}" + (f" — {detail}" if detail and not condition else ""))
    return condition


def set_env(**kwargs):
    for key in KEYS:
        os.environ.pop(key, None)
    for key, value in kwargs.items():
        if value is not None:
            os.environ[key] = value


def make_audio_file() -> str:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_t_audio.mp3")
    with open(path, "wb") as f:
        f.write(b"fake audio")
    return path


def main() -> int:
    results = []

    # --- определение провайдера ---
    set_env()
    results.append(check("без ключей провайдера нет", agent_module.active_provider() == "none"))

    set_env(HF_TOKEN="t")
    results.append(check("только HF_TOKEN -> huggingface", agent_module.active_provider() == "huggingface"))

    set_env(GROQ_API_KEY="g")
    results.append(check("только GROQ_API_KEY -> groq", agent_module.active_provider() == "groq"))

    set_env(GROQ_API_KEY="g", HF_TOKEN="t")
    results.append(
        check(
            "при обоих ключах приоритет у Groq",
            agent_module.active_provider() == "groq",
            "если HF-лимит исчерпан, начинать с HF — потеря времени на каждом вопросе",
        )
    )

    # Пустая строка не должна считаться заданным ключом: так выглядит
    # незаполненный секрет в интерфейсе Space.
    set_env(GROQ_API_KEY="", HF_TOKEN="t")
    results.append(
        check("пустой GROQ_API_KEY не считается заданным", agent_module.active_provider() == "huggingface")
    )

    # --- построение модели ---
    set_env(GROQ_API_KEY="g")
    model = agent_module.build_model("openai/gpt-oss-120b", "groq")
    results.append(check("для Groq создаётся OpenAI-совместимая модель", "OpenAI" in type(model).__name__,
                         type(model).__name__))
    results.append(
        check(
            "адрес Groq подставлен верно",
            "api.groq.com" in str(model.client.base_url),
            str(model.client.base_url),
        )
    )
    results.append(check("model_id сохранён", model.model_id == "openai/gpt-oss-120b"))

    set_env(HF_TOKEN="t")
    hf_model = agent_module.build_model("Qwen/Qwen2.5-72B-Instruct", "huggingface")
    results.append(check("для HF создаётся InferenceClientModel",
                         type(hf_model).__name__ == "InferenceClientModel", type(hf_model).__name__))

    # --- список моделей под провайдера ---
    results.append(
        check(
            "у Groq свой список моделей",
            agent_module.GROQ_MODEL_CANDIDATES != agent_module.MODEL_CANDIDATES
            and all("/" in m for m in agent_module.GROQ_MODEL_CANDIDATES),
            str(agent_module.GROQ_MODEL_CANDIDATES),
        )
    )

    # --- без ключей select_working_model должен сразу сказать, чего не хватает ---
    set_env()
    try:
        agent_module.select_working_model(verbose=False)
        results.append(check("без ключей выбрасывается понятная ошибка", False, "исключения не было"))
    except RuntimeError as e:
        text = str(e)
        results.append(check("без ключей выбрасывается ошибка", True))
        results.append(check("в ошибке упомянут GROQ_API_KEY", "GROQ_API_KEY" in text))
        results.append(check("в ошибке есть ссылка на получение ключа", "console.groq.com" in text))

    # --- распознавание речи: приоритет и откат ---
    audio = make_audio_file()
    try:
        asr = tools_module.TranscribeAudioTool()

        set_env(GROQ_API_KEY="g", HF_TOKEN="t")
        order = []
        asr._transcribe_via_groq = lambda p, k: (order.append("groq"), "текст Groq")[1]
        asr._transcribe_via_hf = lambda p: (order.append("hf"), "текст HF")[1]
        out = asr.forward(audio)
        results.append(check("речь: сначала Groq", out == "текст Groq" and order == ["groq"], f"{out} {order}"))

        order.clear()
        def failing_groq(p, k):
            order.append("groq")
            raise RuntimeError("429 rate limit")
        asr._transcribe_via_groq = failing_groq
        out = asr.forward(audio)
        results.append(
            check("речь: при падении Groq откат на HF", out == "текст HF" and order == ["groq", "hf"],
                  f"{out} {order}")
        )

        set_env()
        out = asr.forward(audio)
        results.append(check("речь: без ключей понятное сообщение", "GROQ_API_KEY" in out, out[:80]))

        out = asr.forward(audio + ".нет")
        results.append(check("речь: отсутствующий файл распознан как таковой", "нет по пути" in out, out[:80]))
    finally:
        os.path.exists(audio) and os.remove(audio)

    # --- зрение: приоритет и откат ---
    image = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_t_img.png")
    try:
        from PIL import Image

        Image.new("RGB", (4, 4), "blue").save(image)
        vision = tools_module.DescribeImageTool()

        set_env(GROQ_API_KEY="g", HF_TOKEN="t")
        order = []
        vision._describe_via_groq = lambda p, q, k: (order.append("groq"), "вижу Groq")[1]
        vision._describe_via_hf = lambda p, q: (order.append("hf"), "вижу HF")[1]
        out = vision.forward(image, "что тут?")
        results.append(check("зрение: сначала Groq", out == "вижу Groq" and order == ["groq"], f"{out} {order}"))

        order.clear()
        def failing_vision(p, q, k):
            order.append("groq")
            raise RuntimeError("model not found")
        vision._describe_via_groq = failing_vision
        out = vision.forward(image, "что тут?")
        results.append(
            check("зрение: при падении Groq откат на HF", out == "вижу HF" and order == ["groq", "hf"],
                  f"{out} {order}")
        )

        results.append(
            check("зрение: data-URI собирается корректно",
                  tools_module.DescribeImageTool._data_uri(image).startswith("data:image/png;base64,"))
        )

        # У Groq и HF имена моделей зрения разные — подставлять чужое бессмысленно.
        results.append(
            check("зрение: у Groq своя модель, отличная от HF",
                  vision.GROQ_VISION_MODEL != vision.DEFAULT_VISION_MODEL,
                  f"{vision.GROQ_VISION_MODEL} / {vision.DEFAULT_VISION_MODEL}")
        )
    finally:
        os.path.exists(image) and os.remove(image)

    set_env()
    print("\n" + "-" * 70)
    if all(results):
        print(f"Все {len(results)} проверок прошли.")
        return 0
    print(f"ПРОВАЛЕНО: {results.count(False)} из {len(results)}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
