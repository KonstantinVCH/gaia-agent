"""
Вспомогательные инструменты (Tools) для агента, который проходит
финальное задание Unit 4 (бенчмарк GAIA) курса Hugging Face Agents Course.

Помимо готовых инструментов из smolagents (веб-поиск, чтение веб-страниц),
здесь определены собственные инструменты:
  - DownloadGaiaFileTool   — скачивание файла, приложенного к вопросу (GET /files/{task_id})
  - TranscribeAudioTool    — распознавание речи в аудио через HF Inference API (Whisper)
  - YoutubeTranscriptTool  — получение текстовой расшифровки ролика YouTube по ссылке
  - DescribeImageTool      — «зрение»: описание/анализ картинки через vision-модель (VLM)
  - ReadPdfTool            — извлечение текста из PDF по ссылке или локальному пути
  - WikipediaTool          — Википедия через API, с режимом разбора таблиц
  - ReadTextFileTool       — чтение локального текстового файла (open в песочнице запрещён)

Набор инструментов подобран под реальный вид вопросов из учебного подмножества
GAIA (Unit 4): среди них есть вопросы с приложенными .mp3/.xlsx/.py файлами
и вопросы про содержание конкретных роликов на YouTube — без этих инструментов
агент такие вопросы решить не может в принципе, а не просто отвечает хуже.
"""

import base64
import mimetypes
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
        "Downloads the file attached to a GAIA question by its task_id "
        "(image, spreadsheet, audio, text file, etc.). Returns the local path "
        "to the saved file, which you then pass to another tool: "
        "transcribe_audio for .mp3/.wav, describe_image for .png/.jpg, "
        "read_text_file for .py/.txt, or pandas in your own code for .xlsx/.csv."
    )
    inputs = {
        "task_id": {
            "type": "string",
            "description": "The task_id of the question that has an attached file.",
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
        "Transcribes speech from a local audio file and returns the text. "
        "Takes a file path (e.g. the one returned by download_gaia_file). "
        "Use this for questions like 'listen to the recording and tell me...'."
    )
    inputs = {
        "file_path": {
            "type": "string",
            "description": "Local path to the audio file (.mp3, .wav, etc.).",
        }
    }
    output_type = "string"

    def __init__(self, model: str = "openai/whisper-large-v3"):
        super().__init__()
        self.model = model

    # Groq отдаёт Whisper по OpenAI-совместимому адресу и бесплатно, поэтому
    # при заданном GROQ_API_KEY распознавание идёт туда. Это не украшение:
    # когда месячный лимит Hugging Face исчерпан, HF-путь возвращает ошибку, и
    # без альтернативы все вопросы с аудио теряются целиком.
    GROQ_ASR_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
    GROQ_ASR_MODEL = "whisper-large-v3"

    def _transcribe_via_groq(self, file_path: str, api_key: str) -> str:
        with open(file_path, "rb") as audio:
            response = requests.post(
                self.GROQ_ASR_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                files={"file": (os.path.basename(file_path), audio)},
                data={"model": self.GROQ_ASR_MODEL, "response_format": "text"},
                timeout=180,
            )
        response.raise_for_status()
        # response_format=text отдаёт голый текст, а не JSON.
        return response.text.strip()

    def _transcribe_via_hf(self, file_path: str) -> str:
        from huggingface_hub import InferenceClient

        client = InferenceClient(token=os.getenv("HF_TOKEN"))
        result = client.automatic_speech_recognition(file_path, model=self.model)
        text = getattr(result, "text", None) or (result.get("text") if isinstance(result, dict) else None)
        return text or str(result)

    def forward(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            return f"Файла нет по пути {file_path}. Сначала скачайте его через download_gaia_file."

        groq_key = os.getenv("GROQ_API_KEY")
        errors = []

        # Порядок важен: если ключ Groq задан, значит HF, скорее всего, уже
        # исчерпан — начинать с него было бы потерей времени на каждом вопросе.
        attempts = []
        if groq_key:
            attempts.append(("Groq", lambda: self._transcribe_via_groq(file_path, groq_key)))
        if os.getenv("HF_TOKEN"):
            attempts.append(("Hugging Face", lambda: self._transcribe_via_hf(file_path)))

        if not attempts:
            return (
                "Нет ключа для распознавания речи: задайте GROQ_API_KEY "
                "(бесплатно, без карты) или HF_TOKEN."
            )

        for label, call in attempts:
            try:
                text = call()
                if text:
                    return text
                errors.append(f"{label}: пустой ответ")
            except Exception as e:
                errors.append(f"{label}: {type(e).__name__}: {str(e)[:150]}")

        return f"Не удалось распознать аудио {file_path}. Попытки — " + "; ".join(errors)


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
        "Returns the transcript (subtitles) of a YouTube video from its URL. "
        "Use it to answer questions about what was SAID in the video. "
        "It cannot answer questions about what is VISIBLE on screen."
    )
    inputs = {
        "url_or_video_id": {
            "type": "string",
            "description": "YouTube video URL or bare video_id.",
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


class DescribeImageTool(Tool):
    """
    «Зрение» для агента: отправляет локальную картинку в vision-модель (VLM)
    через Hugging Face Inference API и возвращает текстовый ответ на заданный
    вопрос об изображении.

    Зачем это нужно: в учебном наборе GAIA есть вопросы с приложенными
    картинками (например, шахматная позиция на изображении — нужно назвать
    лучший ход). Без vision-инструмента такой вопрос принципиально нерешаем,
    и агент может только честно признать это.

    Работает по тому же HF_TOKEN, что и основная модель. Модель по умолчанию
    можно переопределить переменной окружения VISION_MODEL_ID — это полезно,
    если конкретная VLM недоступна на вашем аккаунте или тарифе.
    """

    name = "describe_image"
    description = (
        "Looks at a local image and answers a question about it. Takes an image "
        "file path (e.g. the one returned by download_gaia_file) and a question "
        "stating exactly what to look for. Returns the vision model's answer. "
        "Use it for any question that requires seeing an image."
    )
    inputs = {
        "file_path": {
            "type": "string",
            "description": "Local path to the image file (.png, .jpg, etc.).",
        },
        "question": {
            "type": "string",
            "description": (
                "What to determine from the image. Be specific, e.g.: 'This is "
                "a chess position, black to move. Describe every piece placement "
                "in notation and name the best move for black.'"
            ),
        },
    }
    output_type = "string"

    DEFAULT_VISION_MODEL = "Qwen/Qwen2.5-VL-72B-Instruct"

    def __init__(self, model: str | None = None):
        super().__init__()
        self.model = model or os.getenv("VISION_MODEL_ID", self.DEFAULT_VISION_MODEL)

    # Vision через Groq — на случай исчерпанного лимита Hugging Face. Модель
    # другая, потому что имена моделей у провайдеров свои: на Groq картинки
    # принимает qwen3.6-27b, на HF — Qwen2.5-VL. Подставлять сюда имя от
    # другого провайдера бессмысленно, поэтому у каждого пути своё.
    GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
    GROQ_VISION_MODEL = "qwen/qwen3.6-27b"

    @staticmethod
    def _data_uri(file_path: str) -> str:
        """Картинка как data-URI: chat-совместимые API принимают её прямо в сообщении."""
        mime = mimetypes.guess_type(file_path)[0] or "image/png"
        with open(file_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{encoded}"

    @staticmethod
    def _vision_messages(question: str, data_uri: str) -> list:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ]

    def _describe_via_groq(self, file_path: str, question: str, api_key: str) -> str:
        model = os.getenv("GROQ_VISION_MODEL_ID", self.GROQ_VISION_MODEL)
        response = requests.post(
            self.GROQ_CHAT_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": self._vision_messages(question, self._data_uri(file_path)),
                "max_tokens": 800,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _describe_via_hf(self, file_path: str, question: str) -> str:
        from huggingface_hub import InferenceClient

        client = InferenceClient(token=os.getenv("HF_TOKEN"))
        response = client.chat_completion(
            model=self.model,
            messages=self._vision_messages(question, self._data_uri(file_path)),
            max_tokens=800,
        )
        return response.choices[0].message.content

    def forward(self, file_path: str, question: str) -> str:
        if not os.path.exists(file_path):
            return f"Файл не найден: {file_path}"

        groq_key = os.getenv("GROQ_API_KEY")
        attempts = []
        if groq_key:
            attempts.append(("Groq", lambda: self._describe_via_groq(file_path, question, groq_key)))
        if os.getenv("HF_TOKEN"):
            attempts.append(("Hugging Face", lambda: self._describe_via_hf(file_path, question)))

        if not attempts:
            return (
                "Нет ключа для анализа изображений: задайте GROQ_API_KEY "
                "(бесплатно, без карты) или HF_TOKEN."
            )

        errors = []
        for label, call in attempts:
            try:
                text = call()
                if text:
                    return text
                errors.append(f"{label}: пустой ответ")
            except Exception as e:
                errors.append(f"{label}: {type(e).__name__}: {str(e)[:150]}")

        return (
            f"Не удалось проанализировать изображение {file_path}. "
            f"Попытки — {'; '.join(errors)}. Модель зрения можно заменить: "
            f"GROQ_VISION_MODEL_ID (для Groq) или VISION_MODEL_ID (для Hugging Face)."
        )


class ReadPdfTool(Tool):
    """
    Извлекает текст из PDF — по ссылке или из локального файла.

    Зачем нужен отдельный инструмент: готовый visit_webpage из smolagents
    прогоняет ответ сервера через markdownify(response.text), то есть трактует
    его как HTML. Для PDF это даёт бинарный мусор, по которому модель ничего
    не найдёт и, что хуже, может начать выдумывать. При этом в учебном наборе
    GAIA есть вопросы, где ответ лежит именно в научной статье (например,
    номер гранта NASA в разделе Acknowledgements) — без чтения PDF они
    нерешаемы.

    Есть поиск по подстроке: у статьи может быть 30 страниц, а нужен один
    абзац, и отдавать модели весь текст — значит топить нужное в контексте.
    """

    name = "read_pdf"
    description = (
        "Extracts text from a PDF document by URL or local path. Use this for "
        "scientific papers and any .pdf — visit_webpage does NOT work on PDFs, it "
        "returns binary garbage. Optional 'search' argument: pass a substring "
        "(e.g. 'NASA award' or 'Acknowledgements') to get only the fragments "
        "around its occurrences instead of the whole document."
    )
    inputs = {
        "url_or_path": {
            "type": "string",
            "description": "URL of the PDF or path to a local file.",
        },
        "search": {
            "type": "string",
            "description": (
                "Optional: search for this substring and return only the "
                "fragments around it instead of the whole text. Case-insensitive."
            ),
            "nullable": True,
        },
    }
    output_type = "string"

    MAX_CHARS = 20000
    CONTEXT_CHARS = 1500

    def forward(self, url_or_path: str, search: str | None = None) -> str:
        local_path = url_or_path
        temp_file = None

        # Если это ссылка — сначала скачиваем во временный файл.
        if url_or_path.lower().startswith(("http://", "https://")):
            try:
                resp = requests.get(
                    url_or_path,
                    timeout=45,
                    headers={"User-Agent": "Mozilla/5.0 (compatible; GaiaAgent/1.0)"},
                )
                resp.raise_for_status()
            except requests.exceptions.RequestException as e:
                return f"Не удалось скачать PDF по ссылке {url_or_path}: {e}"

            # Проверяем, что это действительно PDF: сервер мог отдать
            # HTML-страницу (например, страницу-обёртку arXiv или капчу),
            # и молча пытаться разобрать её как PDF — значит получить
            # невнятную ошибку вместо понятной подсказки.
            if not resp.content[:5].startswith(b"%PDF"):
                ctype = resp.headers.get("content-type", "неизвестно")
                return (
                    f"По ссылке {url_or_path} лежит не PDF (Content-Type: {ctype}). "
                    f"Если это страница-обёртка (например, arXiv /abs/), найди на ней "
                    f"прямую ссылку на .pdf через visit_webpage и передай её сюда."
                )

            temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            temp_file.write(resp.content)
            temp_file.close()
            local_path = temp_file.name

        if not os.path.exists(local_path):
            return f"Файл не найден: {local_path}"

        try:
            from pypdf import PdfReader

            reader = PdfReader(local_path)
            pages = []
            for page_number, page in enumerate(reader.pages, start=1):
                try:
                    pages.append((page_number, page.extract_text() or ""))
                except Exception:
                    # Одна битая страница не должна ронять разбор всего документа.
                    pages.append((page_number, ""))
        except Exception as e:
            return f"Не удалось разобрать PDF {url_or_path}: {e}"
        finally:
            if temp_file:
                try:
                    os.unlink(temp_file.name)
                except OSError:
                    pass

        full_text = "\n".join(text for _, text in pages)
        if not full_text.strip():
            return (
                f"PDF открылся ({len(pages)} стр.), но текста в нём нет — "
                f"скорее всего это сканы изображений. Текстовый слой отсутствует."
            )

        if search:
            needle = search.lower()
            fragments = []
            for page_number, text in pages:
                lowered = text.lower()
                start = 0
                while True:
                    position = lowered.find(needle, start)
                    if position == -1:
                        break
                    left = max(0, position - self.CONTEXT_CHARS // 2)
                    right = min(len(text), position + self.CONTEXT_CHARS // 2)
                    fragments.append(f"[стр. {page_number}] ...{text[left:right]}...")
                    start = position + len(needle)
            if fragments:
                joined = "\n\n".join(fragments)
                return f"Найдено вхождений: {len(fragments)} (всего страниц: {len(pages)})\n\n{joined}"[: self.MAX_CHARS]
            return (
                f"Подстрока '{search}' в документе не найдена "
                f"(страниц: {len(pages)}, символов текста: {len(full_text)}). "
                f"Попробуй другую формулировку или запроси документ без параметра search."
            )

        result = f"PDF, страниц: {len(pages)}\n\n{full_text}"
        if len(result) > self.MAX_CHARS:
            result = (
                result[: self.MAX_CHARS]
                + f"\n\n...[текст обрезан; всего символов {len(full_text)}. "
                f"Чтобы найти нужное место, вызови read_pdf с параметром search]"
            )
        return result


class WikipediaTool(Tool):
    """
    Прямой доступ к Википедии через её официальный API — в обход веб-поиска.

    Зачем отдельный инструмент, если есть web_search:
      1. DuckDuckGo на Hugging Face Spaces регулярно упирается в лимиты и
         отвечает отказом. Википедия же отдаёт данные своим API стабильно и
         без ключа, так что это заметно более надёжный путь к тем вопросам,
         ответ на которые лежит именно там (а таких в наборе GAIA много).
      2. Главное: у Википедии ответ часто лежит в ТАБЛИЦЕ — дискография по
         годам, статистика игрока, число атлетов по странам. Обычный текстовый
         пересказ страницы таблицы теряет, а вопросы GAIA требуют именно
         посчитать по строкам. Поэтому есть режим mode='tables', который
         возвращает таблицы страницы как текст, пригодный для подсчёта.

    Замечание о проверке: логика разбора таблиц протестирована на локальном
    образце разметки (см. test_wikipedia.py). Живые запросы к Википедии в
    песочнице, где готовился код, заблокированы прокси, поэтому сетевой путь
    целиком проверяется уже при первом запуске у вас.
    """

    name = "wikipedia"
    description = (
        "Reads a Wikipedia article via the Wikipedia API. PREFER THIS over "
        "web_search whenever the answer is likely on Wikipedia (people, albums, "
        "sports statistics, Olympics, countries): it goes straight to the source "
        "and does not hit search-engine rate limits. "
        "mode='text' (default) returns the article prose. "
        "mode='tables' returns the article's tables as CSV — use it whenever you "
        "must COUNT or FILTER rows (albums by year, player statistics, athletes "
        "per country); the text mode drops tables entirely, so counting is "
        "impossible there. Optional 'search' returns only fragments around a "
        "substring instead of the whole article."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "Article title or search query, e.g. 'Mercedes Sosa discography'.",
        },
        "mode": {
            "type": "string",
            "description": "'text' for article prose, 'tables' for the article tables as CSV.",
            "nullable": True,
        },
        "search": {
            "type": "string",
            "description": (
                "Optional: return only fragments around this substring instead of "
                "the whole article. Case-insensitive."
            ),
            "nullable": True,
        },
    }
    output_type = "string"

    API_URL = "https://en.wikipedia.org/w/api.php"
    MAX_CHARS = 20000
    CONTEXT_CHARS = 1200
    HEADERS = {"User-Agent": "GaiaAgent/1.0 (HF Agents Course exercise)"}

    def _resolve_title(self, query: str) -> str | None:
        """Ищет статью по запросу и возвращает точное название первой найденной."""
        resp = requests.get(
            self.API_URL,
            params={"action": "query", "list": "search", "srsearch": query, "srlimit": 1, "format": "json"},
            headers=self.HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        hits = resp.json().get("query", {}).get("search", [])
        return hits[0]["title"] if hits else None

    @staticmethod
    def _extract_fragments(text: str, needle: str, context: int) -> list[str]:
        lowered = text.lower()
        needle = needle.lower()
        fragments, start = [], 0
        while True:
            position = lowered.find(needle, start)
            if position == -1:
                break
            left = max(0, position - context // 2)
            right = min(len(text), position + context // 2)
            fragments.append(f"...{text[left:right]}...")
            start = position + len(needle)
        return fragments

    @staticmethod
    def _tables_from_html(html: str) -> str:
        """
        Достаёт таблицы из HTML статьи и отдаёт их как CSV-текст.

        pandas.read_html сам находит все таблицы; CSV выбран потому, что он
        компактнее markdown и модели проще посчитать по нему строки.
        """
        import io

        import pandas as pd

        # flavor='lxml' указан явно и намеренно. Без него pandas при неудаче
        # откатывается на связку bs4+html5lib, и если html5lib не установлен —
        # вместо понятного «таблиц нет» вылетает ImportError про отсутствующий
        # пакет. Это ровно тот случай, когда неявное поведение библиотеки
        # превращает штатную ситуацию в непонятную поломку.
        try:
            tables = pd.read_html(io.StringIO(html), flavor="lxml")
        except ValueError:
            # read_html бросает ValueError, когда таблиц в разметке нет вовсе.
            return "В статье не найдено таблиц."
        if not tables:
            return "В статье не найдено таблиц."

        parts = [f"Найдено таблиц: {len(tables)}"]
        for index, table in enumerate(tables, start=1):
            # Плоские заголовки: у википедийных таблиц часто многоуровневые,
            # а в CSV они превращаются в кашу из кортежей.
            if isinstance(table.columns, pd.MultiIndex):
                table.columns = [" ".join(str(p) for p in col if str(p) != "nan") for col in table.columns]
            parts.append(f"\n=== Таблица {index} (строк: {len(table)}) ===\n{table.to_csv(index=False)}")
        return "\n".join(parts)

    def forward(self, query: str, mode: str | None = None, search: str | None = None) -> str:
        # Опечатку в значении режима трактуем в пользу таблиц: раньше любое
        # значение кроме точного "tables" молча уходило в текстовую ветку,
        # и модель, попросившая mode='table', получала прозу без таблиц,
        # решала, что данных нет, и отвечала наугад. Молчаливый откат к
        # другому поведению — худший вид ошибки: он выглядит как успех.
        mode = (mode or "text").strip().lower()
        if mode.startswith("tab"):
            mode = "tables"

        try:
            title = self._resolve_title(query)
        except requests.exceptions.RequestException as e:
            return f"Не удалось обратиться к API Википедии: {e}"
        if not title:
            return f"В Википедии не найдено статьи по запросу '{query}'."

        try:
            if mode == "tables":
                # action=parse отдаёт готовый HTML статьи — из него и берём таблицы.
                resp = requests.get(
                    self.API_URL,
                    params={"action": "parse", "page": title, "prop": "text", "format": "json"},
                    headers=self.HEADERS,
                    timeout=45,
                )
                resp.raise_for_status()
                html = resp.json()["parse"]["text"]["*"]
                body = self._tables_from_html(html)
            else:
                resp = requests.get(
                    self.API_URL,
                    params={
                        "action": "query",
                        "prop": "extracts",
                        "explaintext": 1,
                        "titles": title,
                        "format": "json",
                    },
                    headers=self.HEADERS,
                    timeout=45,
                )
                resp.raise_for_status()
                pages = resp.json()["query"]["pages"]
                body = next(iter(pages.values())).get("extract", "")
                if not body.strip():
                    return f"Статья '{title}' найдена, но текст пуст."
        except requests.exceptions.RequestException as e:
            return f"Ошибка запроса к Википедии для статьи '{title}': {e}"
        except Exception as e:
            return f"Не удалось разобрать ответ Википедии для статьи '{title}': {e}"

        header = f"Статья: {title}\nСсылка: https://en.wikipedia.org/wiki/{title.replace(' ', '_')}\n"

        if search:
            fragments = self._extract_fragments(body, search, self.CONTEXT_CHARS)
            if fragments:
                joined = "\n\n".join(fragments)
                return f"{header}Найдено вхождений '{search}': {len(fragments)}\n\n{joined}"[: self.MAX_CHARS]
            return (
                f"{header}Подстрока '{search}' в статье не найдена. "
                f"Попробуй другую формулировку, режим mode='tables' "
                f"(если ответ в таблице) или запрос без параметра search."
            )

        result = header + "\n" + body
        if len(result) > self.MAX_CHARS:
            result = result[: self.MAX_CHARS] + "\n\n...[обрезано; уточни запрос параметром search]"
        return result


class ReadTextFileTool(Tool):
    """
    Читает локальный текстовый файл и возвращает его содержимое.

    Инструмент необходим из-за устройства песочницы smolagents: встроенный
    интерпретатор Python намеренно запрещает `open` (его нет среди разрешённых
    builtins), поэтому агент физически не может прочитать скачанный файл
    кодом — ни через open, ни через pathlib. До появления этого инструмента
    указание в промпте «прочитай текст файла» было невыполнимым: агент
    упирался в InterpreterError и тратил все шаги впустую.

    Нужен для вопросов с приложенным `.py`, `.txt`, `.json`, `.md` и подобным.
    Для `.xlsx`/`.csv` он не требуется: их агент читает через pandas, которому
    доступ к файлам разрешён.
    """

    name = "read_text_file"
    description = (
        "Reads a local text file (.py, .txt, .json, .md, .csv, etc.) and returns "
        "its contents as a string. Takes a path — for example the one returned by "
        "download_gaia_file. Use this INSTEAD of open() in your code: open() is "
        "forbidden in the sandbox and will fail."
    )
    inputs = {
        "file_path": {
            "type": "string",
            "description": "Local path to the text file.",
        },
        "max_chars": {
            "type": "integer",
            "description": "Optional: maximum number of characters to return (default 20000).",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, file_path: str, max_chars: int | None = None) -> str:
        limit = max_chars or 20000
        try:
            # errors='replace': файл может оказаться не в UTF-8, и падать
            # из-за одного битого байта, потеряв весь остальной текст, глупо.
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(limit + 1)
        except FileNotFoundError:
            return f"Файл не найден: {file_path}"
        except IsADirectoryError:
            return f"Это папка, а не файл: {file_path}"
        except OSError as e:
            return f"Не удалось прочитать файл {file_path}: {e}"

        if not content.strip():
            return f"Файл {file_path} пуст."
        if len(content) > limit:
            return content[:limit] + f"\n...[обрезано на {limit} символах]"
        return content
