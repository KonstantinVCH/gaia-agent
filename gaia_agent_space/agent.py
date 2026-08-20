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

from smolagents import CodeAgent, InferenceClientModel, DuckDuckGoSearchTool, VisitWebpageTool

from tools import DownloadGaiaFileTool, TranscribeAudioTool, YoutubeTranscriptTool

# Модель можно переопределить переменной окружения MODEL_ID в секретах Space,
# если выбранная модель окажется недоступна на вашем аккаунте/провайдере
# инференса. Qwen2.5-Coder-32B-Instruct — модель, которую в тех же целях
# использовали примеры в самом курсе (Unit 2, Unit 3).
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"

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
    "Формат ответа (важно!): ответь ОДНИМ финальным значением через "
    "инструмент final_answer, без пояснений, без слов вроде 'Ответ:' или "
    "'FINAL ANSWER', без кавычек, если их явно не просят. Если просят число — "
    "верни только число (без единиц измерения и разделителей тысяч, если не "
    "указано иное). Если просят список — верни элементы через запятую, ровно "
    "в том формате, который указан в вопросе."
)


class GaiaAgent:
    """Обёртка над smolagents.CodeAgent, приспособленная под API курса."""

    def __init__(self, api_url: str, model_id: str | None = None):
        self.api_url = api_url
        model_id = model_id or os.getenv("MODEL_ID", DEFAULT_MODEL_ID)

        model = InferenceClientModel(model_id=model_id)

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
        print(f"GaiaAgent initialized with model: {model_id}")

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
            return str(result).strip()
        except Exception as e:
            print(f"GaiaAgent error on task {task_id}: {e}")
            return "Не удалось получить ответ (ошибка агента)."
