"""
24 проверки работоспособности окружения перед деплоем Space.
"""

import sys
import os

CHECKS_PASSED = 0
CHECKS_FAILED = []


def check(name, fn):
    global CHECKS_PASSED
    try:
        fn()
        CHECKS_PASSED += 1
        print(f"  OK  {name}")
    except Exception as e:
        CHECKS_FAILED.append((name, str(e)))
        print(f"  FAIL  {name}: {e}")


# -- 1. Python >= 3.9
check("Python версия >= 3.9", lambda: (
    None if sys.version_info >= (3, 9)
    else (_ for _ in ()).throw(RuntimeError(f"Python {sys.version_info.major}.{sys.version_info.minor}"))
))

# -- 2–10. Импорты зависимостей
check("import gradio", lambda: __import__("gradio"))
check("import requests", lambda: __import__("requests"))
check("import pandas", lambda: __import__("pandas"))
check("import openpyxl", lambda: __import__("openpyxl"))
check("import smolagents", lambda: __import__("smolagents"))
check("import duckduckgo_search", lambda: __import__("duckduckgo_search"))
check("import markdownify", lambda: __import__("markdownify"))
check("import huggingface_hub", lambda: __import__("huggingface_hub"))
check("import youtube_transcript_api", lambda: __import__("youtube_transcript_api"))

# -- 11–14. Наличие файлов Space
HERE = os.path.dirname(os.path.abspath(__file__))

def _file(name):
    path = os.path.join(HERE, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Файл не найден: {path}")

check("agent.py существует", lambda: _file("agent.py"))
check("app.py существует", lambda: _file("app.py"))
check("tools.py существует", lambda: _file("tools.py"))
check("requirements.txt существует", lambda: _file("requirements.txt"))

# -- 15–18. Импорты модулей Space
sys.path.insert(0, HERE)
check("from agent import GaiaAgent", lambda: __import__("agent").GaiaAgent)
check("from tools import DownloadGaiaFileTool", lambda: __import__("tools").DownloadGaiaFileTool)
check("from tools import TranscribeAudioTool", lambda: __import__("tools").TranscribeAudioTool)
check("from tools import YoutubeTranscriptTool", lambda: __import__("tools").YoutubeTranscriptTool)

# -- 19–21. Константы agent.py
def _check_constants():
    import agent as a
    assert hasattr(a, "DEFAULT_MODEL_ID") and a.DEFAULT_MODEL_ID, "DEFAULT_MODEL_ID не задан"
    assert hasattr(a, "EXTRA_IMPORTS") and a.EXTRA_IMPORTS, "EXTRA_IMPORTS пустой"
    assert hasattr(a, "ANSWER_INSTRUCTIONS") and a.ANSWER_INSTRUCTIONS, "ANSWER_INSTRUCTIONS пустой"

check("DEFAULT_MODEL_ID определён", lambda: __import__("agent").DEFAULT_MODEL_ID)
check("EXTRA_IMPORTS непустой список", lambda: (
    None if __import__("agent").EXTRA_IMPORTS
    else (_ for _ in ()).throw(RuntimeError("Пустой"))
))
check("ANSWER_INSTRUCTIONS непустой", lambda: (
    None if __import__("agent").ANSWER_INSTRUCTIONS
    else (_ for _ in ()).throw(RuntimeError("Пустой"))
))

# -- 22. API endpoint доступен
def _check_api():
    import requests
    r = requests.get("https://agents-course-unit4-scoring.hf.space/questions", timeout=10)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list) or len(data) == 0:
        raise RuntimeError("Пустой список вопросов")

check("API /questions доступен и возвращает данные", _check_api)

# -- 23. smolagents инструменты создаются без ошибок
def _check_tools():
    from smolagents import DuckDuckGoSearchTool, VisitWebpageTool
    DuckDuckGoSearchTool()
    VisitWebpageTool()

check("DuckDuckGoSearchTool и VisitWebpageTool создаются", _check_tools)

# -- 24. Gradio Blocks строится без ошибок
def _check_gradio():
    import gradio as gr
    with gr.Blocks() as demo:
        gr.Markdown("test")
        gr.Button("test")
    demo.close()

check("Gradio Blocks строится без исключений", _check_gradio)

# --- Итог ---
TOTAL = 24
print()
if CHECKS_FAILED:
    print(f"Провалено {len(CHECKS_FAILED)} из {TOTAL} проверок:")
    for name, err in CHECKS_FAILED:
        print(f"  FAIL {name}: {err}")
    sys.exit(1)
else:
    print(f"Все {TOTAL} проверок прошли.")
