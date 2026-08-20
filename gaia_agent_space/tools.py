"""
Вспомогательные инструменты (Tools) для агента, который проходит
финальное задание Unit 4 (бенчмарк GAIA) курса Hugging Face Agents Course.

Помимо готовых инструментов из smolagents (веб-поиск, чтение веб-страниц),
здесь определены собственные инструменты:
  - DownloadGaiaFileTool   — скачивание файла, приложенного к вопросу (GET /files/{task_id})
  - TranscribeAudioTool    — распознавание речи в аудио через HF Inference API (Whisper)
  - YoutubeTranscriptTool  — получение текстовой расшифровки ролика YouTube по ссылке

Набор инструментов подобран под реальный вид вопросов из учебного подмножества
GAIA (Unit 4): среди них есть вопросы с приложенными .mp3/.xlsx/.py файлами
и вопросы про содержание конкретных роликов на YouTube — без этих инструментов
агент такие вопросы решить не может в принципе, а не просто отвечает хуже.
"""

import os
import re
import tempfile

import requests
from smolagents import Tool


class DownloadGaiaFileTool(Tool):
    """
    Скачивает файл, приложенный к вопросу бенчмарка GAIA, по его task_id,
    и возвращает локальный путь к сохранённому файлу.

    Не у каждого вопроса есть приложение — если файла нет, API вернёт ошибку
    404, и инструмент сообщит об этом агенту явным текстом, чтобы он не
    зависал в ожидании несуществующего файла.
    """

    name = "download_gaia_file"
    description = (
        "Скачивает файл, приложенный к вопросу GAIA, по его task_id "
        "(картинка, таблица, аудио, текстовый файл и т.п.). "
        "Возвращает путь к локально сохранённому файлу, чтобы его можно "
        "было дальше прочитать/обработать другими средствами "
        "(например, кодом на Python внутри агента)."
    )
    inputs = {
        "task_id": {
            "type": "string",
            "description": "task_id вопроса, к которому приложен файл.",
        }
    }
    output_type = "string"

    def __init__(self, api_url: str):
        super().__init__()
        self.api_url = api_url.rstrip("/")

    def forward(self, task_id: str) -> str:
        url = f"{self.api_url}/files/{task_id}"
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 404:
                return f"У вопроса {task_id} нет приложенного файла (404)."
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            return f"Не удалось скачать файл для {task_id}: {e}"

        # Пытаемся определить расширение файла по заголовку Content-Disposition,
        # если не получилось — сохраняем без расширения.
        content_disposition = resp.headers.get("content-disposition", "")
        filename = None
        if "filename=" in content_disposition:
            filename = content_disposition.split("filename=")[-1].strip('"; ')

        suffix = ""
        if filename and "." in filename:
            suffix = "." + filename.rsplit(".", 1)[-1]

        tmp_dir = tempfile.gettempdir()
        local_path = os.path.join(tmp_dir, f"gaia_{task_id}{suffix or ''}")
        with open(local_path, "wb") as f:
            f.write(resp.content)

        return (
            f"Файл сохранён локально по пути: {local_path} "
            f"(исходное имя: {filename or 'неизвестно'})."
        )


class TranscribeAudioTool(Tool):
    """
    Распознаёт речь в аудиофайле (например, .mp3, скачанном через
    DownloadGaiaFileTool) и возвращает текстовую расшифровку.

    Использует Hugging Face Inference API (модель автоматического
    распознавания речи) через тот же HF_TOKEN, что и основная модель агента —
    отдельный токен или локальная установка whisper не нужны.
    """

    name = "transcribe_audio"
    description = (
        "Распознаёт речь в локальном аудиофайле и возвращает текст. "
        "Принимает путь к файлу (например, тот, что вернул download_gaia_file). "
        "Полезно для вопросов вида «прослушай запись и скажи...»."
    )
    inputs = {
        "file_path": {
            "type": "string",
            "description": "Локальный путь к аудиофайлу (.mp3, .wav и т.п.).",
        }
    }
    output_type = "string"

    def __init__(self, model: str = "openai/whisper-large-v3"):
        super().__init__()
        self.model = model

    def forward(self, file_path: str) -> str:
        try:
            from huggingface_hub import InferenceClient

            client = InferenceClient(token=os.getenv("HF_TOKEN"))
            result = client.automatic_speech_recognition(file_path, model=self.model)
            text = getattr(result, "text", None) or (result.get("text") if isinstance(result, dict) else None)
            return text or str(result)
        except Exception as e:
            return f"Не удалось распознать аудио {file_path}: {e}"


class YoutubeTranscriptTool(Tool):
    """
    Возвращает текстовую расшифровку (субтитры/автосубтитры) видео YouTube
    по его URL или video_id — без скачивания самого видео.

    Ограничение: работает только для того, что произносится/написано в
    субтитрах. Вопросы, требующие анализа изображения на видео (например,
    «сколько видов птиц одновременно в кадре»), эта расшифровка не решает —
    для этого нужен отдельный видео-VLM, которого в этом агенте нет.
    """

    name = "get_youtube_transcript"
    description = (
        "Возвращает текстовую расшифровку (субтитры) видео YouTube по ссылке. "
        "Помогает ответить на вопросы о том, что было СКАЗАНО в видео. "
        "Не подходит для вопросов о том, что видно на экране (визуальный анализ)."
    )
    inputs = {
        "url_or_video_id": {
            "type": "string",
            "description": "Ссылка на видео YouTube или его video_id.",
        }
    }
    output_type = "string"

    @staticmethod
    def _extract_video_id(value: str) -> str:
        match = re.search(r"(?:v=|youtu\.be/|embed/)([A-Za-z0-9_-]{11})", value)
        return match.group(1) if match else value

    def forward(self, url_or_video_id: str) -> str:
        try:
            # В youtube-transcript-api>=1.0 API стал объектным: сначала создаём
            # клиент, затем вызываем fetch() на конкретном video_id (раньше был
            # статический метод get_transcript — в новых версиях он удалён).
            from youtube_transcript_api import YouTubeTranscriptApi

            video_id = self._extract_video_id(url_or_video_id)
            api = YouTubeTranscriptApi()
            fetched = api.fetch(video_id, languages=["en", "ru"])
            text = " ".join(snippet.text for snippet in fetched)
            return text[:8000]  # ограничиваем размер, чтобы не раздувать контекст модели
        except Exception as e:
            return f"Не удалось получить расшифровку для {url_or_video_id}: {e}"
