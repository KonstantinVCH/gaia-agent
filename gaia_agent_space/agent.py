"""
GaiaAgent — агент на базе smolagents.CodeAgent для финального задания
Unit 4 курса Hugging Face Agents Course (тест на подмножестве бенчмарка GAIA).

Идея простая: CodeAgent пишет и исполняет код на Python, чтобы решить задачу,
при необходимости вызывая инструменты (веб-поиск, чтение страниц, скачивание
приложенного к вопросу файла). Ответ, который агент передаёт в свой
внутренний инструмент final_answer(...), и есть то, что уходит на сервер
оценки — поэтому агент явно проинструктирован отвечать только "голым"
финальным значением, без пояснений и без служебных фраз вроде
"FINAL ANSWER:" (сервер сверяет ответ дословно, по точному совпадению).
"""

import os

from smolagents import CodeAgent, DuckDuckGoSearchTool, VisitWebpageTool

from tools import DownloadGaiaFileTool, TranscribeAudioTool, YoutubeTranscriptTool

# --- Выбор провайдера ---
# Приоритет:
#   1. GROQ_API_KEY → Groq (бесплатно, llama-3.3-70b-versatile)
#   2. HF_TOKEN    → HF Inference API (бесплатные лимиты)
# Модель можно переопределить MODEL_ID (секрет Space).

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_HF_MODEL   = "Qwen/Qwen2.5-Coder-32B-Instruct"


def _build_model():
    model_id = os.getenv("MODEL_ID")
    groq_key  = os.getenv("GROQ_API_KEY")
    hf_token  = os.getenv("HF_TOKEN")

    if groq_key:
        from smolagents import OpenAIServerModel
        mid = model_id or DEFAULT_GROQ_MODEL
        print(f"Provider: Groq  |  Model: {mid}")
        return OpenAIServerModel(
            model_id=mid,
            api_base="https://api.groq.com/openai/v1",
            api_key=groq_key,
        ), mid

    from smolagents import InferenceClientModel
    mid = model_id or DEFAULT_HF_MODEL
    print(f"Provider: HF Inference API  |  Model: {mid}")
    return InferenceClientModel(model_id=mid), mid


# Для обратной совместимости (test_space_startup.py проверяет наличие константы)
DEFAULT_MODEL_ID = DEFAULT_GROQ_MODEL

# Библиотеки, которые CodeAgent разрешено импортировать в своём коде —
# нужны, чтобы агент мог сам разобрать вложения вопросов GAIA (таблицы
# Excel/CSV, числа, даты, работу со строками) внутри написанного им же кода.
# По умолчанию интерпретатор smolagents намеренно ограничен в целях
# безопасности, поэтому список даём явно, а не разрешаем всё подряд.
EXTRA_IMPORTS = [
    "pandas", "numpy", "math", "re", "json", "csv", "datetime",
    "collections", "itertools", "statistics", "openpyxl",
]

ANSWER_INSTRUCTIONS = (
    "\n\n---\n"
    "Answer format (important!): provide ONE final value via the final_answer tool, "
    "no explanations, no prefixes like 'Answer:' or 'FINAL ANSWER:', no quotes unless "
    "explicitly requested. If a number is requested — return only the number (no units, "
    "no thousand separators unless specified). If a list is requested — return items "
    "comma-separated exactly as the question specifies."
)


def normalize_answer(raw: str) -> str:
    """Strip common agent prefixes/artifacts before submitting to the scoring server."""
    import re
    s = raw.strip()
    # Remove FINAL ANSWER: prefix
    s = re.sub(r"(?i)^(final answer\s*:\s*|answer\s*:\s*)", "", s).strip()
    # Remove surrounding markdown bold/italic
    s = re.sub(r"^\*+|\*+$", "", s).strip()
    # Remove trailing period only when the value isn't a sentence
    if s.endswith(".") and " " not in s:
        s = s[:-1]
    # Convert Python list repr to comma-separated
    if s.startswith("[") and s.endswith("]"):
        try:
            import ast
            items = ast.literal_eval(s)
            if isinstance(items, list):
                s = ", ".join(str(i).strip().strip("'\"") for i in items)
        except Exception:
            pass
    return s


class GaiaAgent:
    """Обёртка над smolagents.CodeAgent, приспособленная под API курса."""

    def __init__(self, api_url: str, model_id: str | None = None):
        self.api_url = api_url
        model, model_id = _build_model()

        self.agent = CodeAgent(
            tools=[
                DuckDuckGoSearchTool(),
                VisitWebpageTool(),
                DownloadGaiaFileTool(api_url=api_url),
                TranscribeAudioTool(),
                YoutubeTranscriptTool(),
            ],
            model=model,
            add_base_tools=True,   # добавляет PythonInterpreterTool и др. из коробки
            additional_authorized_imports=EXTRA_IMPORTS,
            planning_interval=3,
            max_steps=12,
        )
        print(f"GaiaAgent ready.")

    def __call__(self, question: str, task_id: str | None = None, file_name: str | None = None) -> str:
        prompt = question
        if file_name:
            prompt += (
                f"\n\n(У этого вопроса есть приложенный файл: {file_name}. "
                f"Сначала вызови download_gaia_file с task_id='{task_id}', "
                f"чтобы скачать его локально. Дальше в зависимости от типа файла: "
                f".mp3/.wav — передай локальный путь в transcribe_audio; "
                f".xlsx/.csv — прочитай через pandas прямо в своём коде; "
                f".py — прочитай текст файла и посчитай/выполни то, что в нём написано; "
                f".png/.jpg — учти, что у этого агента нет зрения (vision), "
                f"в таком случае честно верни свою лучшую оценку, а не выдумывай факт.)"
            )
        if "youtube.com" in question or "youtu.be" in question:
            prompt += (
                "\n\n(В вопросе есть ссылка на YouTube-видео. Если нужно понять, "
                "что там ПРОИЗНЕСЕНО, используй get_youtube_transcript. Учти: этот "
                "инструмент не показывает изображение — если вопрос про то, что "
                "ВИДНО в кадре, у агента нет средств это проверить.)"
            )
        prompt += ANSWER_INSTRUCTIONS

        try:
            result = self.agent.run(prompt)
            return normalize_answer(str(result))
        except Exception as e:
            print(f"GaiaAgent error on task {task_id}: {e}")
            return "ERROR"
