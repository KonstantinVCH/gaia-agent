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
import re

from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    InferenceClientModel,
    OpenAIServerModel,
    VisitWebpageTool,
)

from tools import (
    DescribeImageTool,
    DownloadGaiaFileTool,
    ReadPdfTool,
    ReadTextFileTool,
    TranscribeAudioTool,
    WikipediaTool,
    YoutubeTranscriptTool,
)

# ---------------------------------------------------------------------------
# Выбор провайдера инференса
# ---------------------------------------------------------------------------
# Исходно агент умел работать только через Hugging Face Inference API. На
# практике это оказалось узким местом: бесплатный месячный лимит HF
# исчерпывается, и тогда агент не может ответить ни на один вопрос — при
# полностью работоспособном коде. Поэтому провайдер выбирается по тому, какой
# ключ фактически задан в окружении.
#
# Groq выбран как основной альтернативный путь по двум причинам: он бесплатен
# без привязки карты и, что важнее, покрывает все три нужные агенту
# способности — чат, распознавание речи (Whisper) и зрение. Заводить разные
# сервисы под каждую из них не пришлось.
#
# Приоритет у Groq, если задан GROQ_API_KEY: раз пользователь его прописал,
# значит HF-лимит уже исчерпан и пробовать HF первым бессмысленно.
GROQ_API_BASE = "https://api.groq.com/openai/v1"

# Модели Groq. Первой идёт gpt-oss-120b: из production-моделей Groq это самая
# крупная, а вопросы GAIA требуют многоходового рассуждения, где размер модели
# заметно важнее скорости. 20b оставлена запасной — если 120b упрётся в лимит
# частоты запросов, лучше отвечать медленнее и хуже, чем не отвечать вовсе.
GROQ_MODEL_CANDIDATES = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "groq/compound",
]


def _groq_key() -> str | None:
    return os.getenv("GROQ_API_KEY") or None


def active_provider() -> str:
    """Какой провайдер будет использован при текущем окружении."""
    if _groq_key():
        return "groq"
    if os.getenv("HF_TOKEN"):
        return "huggingface"
    return "none"


# Модель — самая хрупкая часть всей конструкции: если она недоступна на вашем
# токене, агент не ответит НИ НА ОДИН вопрос, и сертификата не будет. Поэтому
# модель не задаётся одной жёсткой строкой, а выбирается перебором кандидатов
# с реальной проверкой: список моделей, доступных через Inference Providers,
# со временем меняется, и захардкоженное имя однажды просто перестаёт работать.
#
# Порядок списка осмысленный:
#   1. То, что вы задали сами в MODEL_ID — ваш выбор приоритетнее любых догадок.
#   2. Модель по умолчанию из самого smolagents: библиотека держит там то, что
#      актуально сейчас, так что это лучший ориентир, чем список из курса.
#   3. Дальше — распространённые instruct-модели как запас.
#
# Qwen2.5-Coder-32B-Instruct (модель из примеров курса) оставлена в конце
# списка: она может быть уже не обслуживаемой, но если работает — тоже годится.
def _smolagents_default_model() -> str | None:
    """Модель по умолчанию из установленной версии smolagents, если её видно."""
    try:
        import inspect

        default = inspect.signature(InferenceClientModel.__init__).parameters["model_id"].default
        return default if isinstance(default, str) and default else None
    except Exception:
        return None


MODEL_CANDIDATES = [
    _smolagents_default_model(),
    "Qwen/Qwen3-Next-80B-A3B-Thinking",
    "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
]

# Первый кандидат — для обратной совместимости с прежним поведением и для
# сообщений в логах.
DEFAULT_MODEL_ID = next((m for m in MODEL_CANDIDATES if m), "Qwen/Qwen2.5-Coder-32B-Instruct")

# Библиотеки, которые CodeAgent разрешено импортировать в своём коде —
# нужны, чтобы агент мог сам разобрать вложения вопросов GAIA (таблицы
# Excel/CSV, числа, даты, работу со строками) внутри написанного им же кода.
# По умолчанию интерпретатор smolagents намеренно ограничен в целях
# безопасности, поэтому список даём явно, а не разрешаем всё подряд.
EXTRA_IMPORTS = [
    "pandas", "numpy", "math", "re", "json", "csv", "datetime",
    "collections", "itertools", "statistics", "openpyxl",
]

# Правила ниже — это официальные требования GAIA к формату ответа
# (подтверждено по материалам курса и по эталонным промптам GAIA).
# Курс отдельно оговаривает: в отправляемом ответе НЕ должно быть текста
# "FINAL ANSWER" — только само значение и ничего больше.
#
# Осознанно не включена спорная формулировка исходного промпта GAIA про
# "write the digits in plain text": в разных воспроизведениях эталонного
# промпта она трактуется противоположно (цифрами или словами), и указание
# наугад способно испортить как раз те ответы, которые иначе были бы верны.
# Вместо этого модели велено следовать формату, заданному в самом вопросе.
# Инструкции о формате даны ПО-АНГЛИЙСКИ намеренно, хотя остальные пояснения
# в этом файле русские. Вопросы GAIA и эталонные ответы — англоязычные, а сверка
# дословная: русскоязычная инструкция повышает риск получить «Санкт-Петербург»
# там, где сервер ждёт «Saint Petersburg». Язык требований совпадает с языком
# ожидаемого ответа — так модель не сбивается.
ANSWER_INSTRUCTIONS = (
    "\n\n---\n"
    "ANSWER FORMAT (critical — the server compares answers verbatim):\n"
    "Return ONE final value via the final_answer tool. Answer in English. "
    "No explanations, no reasoning, no prefixes like 'Answer:' or 'FINAL ANSWER', "
    "no quotes and no markdown formatting unless explicitly requested.\n"
    "- Pass a STRING to final_answer, already formatted exactly as asked. Do not "
    "pass a raw computation result and expect it to be formatted for you: if the "
    "question asks for two decimal places, format it yourself (e.g. '89706.00').\n"
    "- If the answer is a NUMBER: digits only. No thousands separators (write 1234, "
    "not 1,234) and no units or symbols ($, %, kg) unless the question explicitly "
    "asks for them.\n"
    "- If the answer is a STRING: no articles (the, a) and no abbreviations — write "
    "cities, organisations and names in full (e.g. 'Saint Petersburg', not "
    "'St. Petersburg').\n"
    "- If the answer is a LIST: comma-separated, applying the rules above to each "
    "element depending on whether it is a number or a string.\n"
    "- If the question specifies a particular answer format, that takes precedence "
    "over these rules — follow it literally."
)


# Указание против «поискового цикла». На реальном прогоне модель на вопросе о
# дискографии сделала девять шагов из двенадцати, каждый раз формулируя новый
# поисковый запрос вместо того, чтобы открыть найденную страницу. Это типичный
# способ исчерпать лимит шагов, ничего не ответив, поэтому запрет на повтор
# даётся явно, вместе с указанием, что делать вместо повтора.
EFFICIENCY_INSTRUCTIONS = (
    "\n\n---\n"
    "HOW TO WORK EFFICIENTLY (you have a limited number of steps):\n"
    "- NEVER repeat a search you have already run, and never run a slight "
    "rewording of it. If a search returned results, the next step is to OPEN "
    "the most promising result (visit_webpage) or to extract the answer from "
    "what you already have — not to search again.\n"
    "- If the answer is likely on Wikipedia, call the wikipedia tool FIRST "
    "instead of web_search. For anything that must be counted or filtered "
    "(albums per year, statistics, athletes per country) call it with "
    "mode='tables': the text mode drops tables, so counting is impossible.\n"
    "- Keep a running note of what you already know in your reasoning. Before "
    "each new tool call, ask yourself: do I already have what I need?\n"
    "- If two different approaches have failed, do not try a third variation "
    "of the same idea — give your best supported answer from the evidence "
    "gathered so far. A well-grounded answer beats running out of steps."
)


def build_model(model_id: str, provider: str | None = None):
    """
    Создаёт объект модели под выбранного провайдера.

    Для Groq используется OpenAIServerModel: Groq отдаёт OpenAI-совместимый
    API, поэтому отдельного класса не нужно — достаточно подменить базовый
    адрес. Это же и причина, по которой Groq удобен как замена: код агента
    остаётся тем же, меняется только конструирование модели.
    """
    provider = provider or active_provider()

    if provider == "groq":
        return OpenAIServerModel(
            model_id=model_id,
            api_base=GROQ_API_BASE,
            api_key=_groq_key(),
        )
    return InferenceClientModel(model_id=model_id)


def select_working_model(candidates: list[str] | None = None, verbose: bool = True):
    """
    Возвращает первую модель из списка, которая реально отвечает.

    Зачем проверять, а не просто взять первую: недоступность модели выясняется
    только при обращении к API. Без проверки прогон падал бы на первом же
    вопросе, потратив время впустую и не дав понять, что причина именно в
    модели. Проверочный запрос делается коротким намеренно — он нужен лишь
    чтобы отличить «модель отвечает» от «модели нет».

    Список кандидатов дедуплицируется с сохранением порядка: MODEL_ID может
    совпасть с одним из встроенных вариантов, и проверять его дважды незачем.
    """
    provider = active_provider()
    if provider == "none":
        raise RuntimeError(
            "Не задан ни один ключ. Нужен либо GROQ_API_KEY (бесплатно, без "
            "карты: https://console.groq.com/keys), либо HF_TOKEN. Если "
            "месячный лимит Hugging Face исчерпан — используйте GROQ_API_KEY."
        )

    requested = os.getenv("MODEL_ID")
    default_list = candidates or (GROQ_MODEL_CANDIDATES if provider == "groq" else MODEL_CANDIDATES)
    raw = ([requested] if requested else []) + default_list

    ordered, seen = [], set()
    for name in raw:
        if name and name not in seen:
            seen.add(name)
            ordered.append(name)

    errors = []
    for model_id in ordered:
        try:
            model = build_model(model_id, provider)
            # Короткий проверочный запрос: важен сам факт ответа, не его содержимое.
            model([{"role": "user", "content": "ping"}], max_tokens=1)
            if verbose:
                print(f"[модель] выбрана: {model_id} (провайдер: {provider})")
            return model, model_id
        except Exception as e:
            reason = f"{type(e).__name__}: {str(e)[:160]}"
            errors.append((model_id, reason))
            if verbose:
                print(f"[модель] {model_id} недоступна — {reason}")

    details = "\n".join(f"  - {m}: {r}" for m, r in errors)
    if provider == "groq":
        hint = (
            "Проверьте, что GROQ_API_KEY верный и не исчерпан лимит: "
            "https://console.groq.com/keys . Конкретную модель можно задать "
            "через MODEL_ID, список доступных: "
            "https://console.groq.com/docs/models"
        )
    else:
        hint = (
            "Обычные причины: истёк HF_TOKEN или исчерпан месячный лимит "
            "Hugging Face. Если лимит исчерпан — задайте GROQ_API_KEY "
            "(бесплатно, без карты): https://console.groq.com/keys . "
            "Либо укажите доступную вам модель через MODEL_ID, список: "
            "https://huggingface.co/inference/models"
        )
    raise RuntimeError(
        f"Ни одна модель не отвечает (провайдер: {provider}). {hint}\n"
        f"Что именно не сработало:\n{details}"
    )


def _format_scalar(value) -> str:
    """
    Превращает нестроковое финальное значение в строку так, как ждёт сервер.

    Отдельная функция нужна потому, что CodeAgent пишет код, и естественный
    для него ход — вернуть результат вычисления как есть: число из pandas,
    список из фильтрации. До этой обработки такие значения уходили на сервер
    в питоновском виде — то есть список превращался в "['broccoli', 'celery']"
    и гарантированно не совпадал с ожидаемым "broccoli, celery".
    """
    import numbers

    # bool — тоже numbers.Number, но "True"/"False" осмысленнее, чем "1"/"0".
    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, numbers.Number):
        # Целое по значению float ("3.0") приводим к "3": счётные вопросы
        # ждут целое, а модель часто отдаёт результат деления или суммы.
        # Дробную часть НЕ трогаем: там, где вопрос требует два знака после
        # запятой, округлять за модель нельзя — это её работа, и промпт
        # требует от неё именно строку в заданном формате.
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    return str(value)


def normalize_answer(raw) -> str:
    """
    Приводит ответ модели к «голому» виду для дословного сравнения на сервере.

    Сервер оценки GAIA сверяет ответ по точному совпадению строк, поэтому
    лишняя точка в конце, префикс «FINAL ANSWER:» или питоновский repr списка
    превращают правильный по смыслу ответ в неправильный формально.
    Инструкции в промпте это уменьшают, но не устраняют: модели регулярно
    добавляют пояснения, а CodeAgent вообще возвращает не строки.

    Сознательно НЕ трогаем регистр и внутреннюю пунктуацию: в GAIA есть
    вопросы, где ответ — имя собственное или строка в заданном формате,
    и «умная» правка там навредит больше, чем поможет.
    """
    if raw is None:
        return ""

    # 0. Нестроковые типы обрабатываем ДО str(): именно здесь терялись
    #    ответы-списки, которых в наборе GAIA несколько.
    if isinstance(raw, (list, tuple, set)):
        items = list(raw)
        if isinstance(raw, set):
            # У множества нет порядка, а сервер сверяет строку дословно.
            # Сортируем, чтобы результат был хотя бы воспроизводимым.
            items = sorted(items, key=lambda x: str(x))
        return ", ".join(_format_scalar(item) for item in items)

    if not isinstance(raw, str):
        text = _format_scalar(raw).strip()
    else:
        text = raw.strip()

    # 1. Снимаем markdown-заборчик, если модель обернула значение в блок кода.
    #    Делать это надо ДО выбора последней строки: иначе последней строкой
    #    окажется сам заборчик, и ответ выродится в пустую строку — то есть
    #    вопрос будет потерян там, где значение было верным.
    fence = re.match(r"^```[a-zA-Z0-9_+-]*\s*\n(.*?)\n?```$", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1).strip()

    # 2. Срезаем служебные префиксы, если модель их всё-таки добавила.
    #
    #    Тонкое место. Жёсткая привязка к началу строки ломает частый случай
    #    «The answer: 42» (префикс не в начале). Полное отсутствие привязки
    #    ломает случай, когда слово answer — часть самого ответа: в
    #    'The word is "answer:" itself' срезалось бы полезное содержимое.
    #    Поэтому срезаем где угодно, но НЕ там, где префикс стоит в кавычках:
    #    кавычка перед ним — верный признак, что это часть текста ответа.
    prefix_pattern = re.compile(
        r"(?:final\s*answer|answer|ответ)\s*[:\-—]\s*",
        flags=re.IGNORECASE,
    )
    for match in reversed(list(prefix_pattern.finditer(text))):
        preceding = text[: match.start()].rstrip()
        if preceding and preceding[-1] in "\"'«“":
            continue  # префикс внутри кавычек — это содержимое, не служебное слово
        tail = text[match.end():].strip()
        if tail:  # пустой хвост означал бы, что мы срезали весь ответ
            text = tail
        break

    # 3. Если после этого осталось несколько строк — берём последнюю
    #    непустую (финальное значение обычно в конце).
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) > 1:
        text = lines[-1]

    # 4. Снимаем обрамляющие кавычки/звёздочки Markdown.
    text = text.strip().strip("`").strip()
    text = re.sub(r"^\*+|\*+$", "", text).strip()
    for quote_pair in (('"', '"'), ("'", "'"), ("«", "»"), ("“", "”")):
        if len(text) >= 2 and text.startswith(quote_pair[0]) and text.endswith(quote_pair[1]):
            text = text[1:-1].strip()

    # 5. Убираем точку в самом конце — но не у сокращений вида "U.S.A.",
    #    где точка часть самого ответа.
    #    Десятичные дроби здесь защищать не нужно и НЕЛЬЗЯ: "3.14" не
    #    заканчивается точкой и в эту ветку не попадает вовсе, а вот "42."
    #    попадает — и точку у него снять надо.
    if text.endswith(".") and not re.search(r"\b[A-Za-z]\.(?:[A-Za-z]\.)+$", text):
        text = text[:-1].strip()

    # 6. Разделители тысяч убираем только там, где это ОДНОЗНАЧНО число.
    #
    #    Здесь была ошибка: шаблон \d{1,3}(?:[,\s]\d{3})+ совпадает и с
    #    "1,234" (число), и с "132,133,134" (список номеров страниц через
    #    запятую — прямо просимый в одном из вопросов формат). Список
    #    склеивался в одно число, то есть верный ответ превращался в мусор.
    #
    #    Различить их в общем случае нельзя: "132,133" одинаково похоже на
    #    оба варианта. Поэтому чистим только когда первая группа короче трёх
    #    цифр ("1,234", "12,345,678") — в списке номера обычно одной ширины,
    #    и такой вид для него нехарактерен. Неоднозначные случаи оставляем
    #    как есть: за них отвечает инструкция модели не ставить разделители.
    #    Пробел как разделитель однозначен — списки пишут через запятую.
    if re.fullmatch(r"-?\d{1,2}(?:,\d{3})+", text) or re.fullmatch(r"-?\d{1,3}(?:\s\d{3})+", text):
        text = re.sub(r"[,\s]", "", text)

    return text.strip()


class GaiaAgent:
    """Обёртка над smolagents.CodeAgent, приспособленная под API курса."""

    def __init__(self, api_url: str, model_id: str | None = None):
        self.api_url = api_url

        if model_id:
            # Явно переданную модель уважаем без перебора.
            model = InferenceClientModel(model_id=model_id)
        else:
            model, model_id = select_working_model()
        self.model_id = model_id

        self.agent = CodeAgent(
            tools=[
                DuckDuckGoSearchTool(),
                VisitWebpageTool(),
                DownloadGaiaFileTool(api_url=api_url),
                TranscribeAudioTool(),
                YoutubeTranscriptTool(),
                DescribeImageTool(),
                ReadPdfTool(),
                ReadTextFileTool(),
                WikipediaTool(),
            ],
            model=model,
            add_base_tools=True,   # базовые инструменты smolagents (для CodeAgent
            # PythonInterpreterTool среди них НЕ появляется: код он и так исполняет сам)
            additional_authorized_imports=EXTRA_IMPORTS,
            planning_interval=5,
            max_steps=16,
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
                f".py/.txt/.json — прочитай его инструментом read_text_file (open() в песочнице запрещён), затем посчитай/выполни то, что там написано; "
                f".png/.jpg — передай локальный путь в describe_image вместе с конкретным "
                f"вопросом о том, что нужно увидеть на картинке. Если describe_image "
                f"вернул ошибку недоступности модели — честно верни свою лучшую оценку, "
                f"а не выдумывай факт.)"
            )
        # Подсказка про PDF: заметная часть «исследовательских» вопросов GAIA
        # упирается в чтение научной статьи, а обычный visit_webpage с PDF
        # выдаёт мусор. Модель об этом сама не догадается — говорим прямо.
        prompt += (
            "\n\n(Если ответ, скорее всего, есть в Википедии — предпочитай инструмент "
            "wikipedia инструменту web_search: он обращается к API напрямую и не "
            "упирается в лимиты поисковика. Если нужно ПОСЧИТАТЬ что-то по строкам "
            "(альбомы по годам, статистика, число участников по странам) — вызывай "
            "wikipedia с mode='tables': в обычном текстовом режиме таблицы теряются.)"
        )

        prompt += (
            "\n\n(Если поиск приведёт к научной статье или любому .pdf — читай его "
            "инструментом read_pdf, а НЕ visit_webpage: visit_webpage разбирает "
            "PDF как HTML и вернёт мусор. Если статья большая, вызывай read_pdf "
            "с параметром search, чтобы найти нужное место: например "
            "search='Acknowledgements' или search='award'.)"
        )

        if "youtube.com" in question or "youtu.be" in question:
            prompt += (
                "\n\n(В вопросе есть ссылка на YouTube-видео. Если нужно понять, "
                "что там ПРОИЗНЕСЕНО, используй get_youtube_transcript. Учти: этот "
                "инструмент не показывает изображение — если вопрос про то, что "
                "ВИДНО в кадре, у агента нет средств это проверить.)"
            )
        prompt += EFFICIENCY_INSTRUCTIONS
        prompt += ANSWER_INSTRUCTIONS

        # Две попытки: сбой у провайдера инференса (таймаут, 503, лимит) —
        # штатная ситуация, и терять из-за неё вопрос целиком обидно.
        # Больше двух не делаем: если модель падает стабильно, повторы только
        # затянут прогон 20 вопросов.
        last_error: Exception | None = None
        for attempt in (1, 2):
            try:
                result = self.agent.run(prompt)
                answer = normalize_answer(result)
                if answer:
                    return answer
                print(f"GaiaAgent: пустой ответ на {task_id} (попытка {attempt})")
            except Exception as e:
                last_error = e
                print(f"GaiaAgent error on task {task_id} (попытка {attempt}): {e}")

        # Отправляем осмысленную заглушку, а не пустую строку: пустой ответ
        # сервер всё равно засчитает как неверный, но по логам будет непонятно,
        # что именно произошло.
        return f"UNKNOWN (agent failed: {type(last_error).__name__ if last_error else 'empty answer'})"
