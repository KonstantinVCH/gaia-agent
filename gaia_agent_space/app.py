import os

import gradio as gr
import requests
import pandas as pd

from agent import GaiaAgent

# (Keep Constants as is)
# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# ----- Собственный агент -----
# BasicAgent из официального шаблона заменён на GaiaAgent — smolagents.CodeAgent
# с шестью инструментами (веб-поиск, чтение страниц, скачивание приложенных
# файлов, распознавание аудио, расшифровка YouTube, анализ изображений).
# Логика самого агента — в agent.py и tools.py.


def run_and_submit_all(profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs GaiaAgent on them, submits all answers,
    and displays the results.

    Это функция-ГЕНЕРАТОР: она делает yield после каждого вопроса, и Gradio
    показывает промежуточный результат сразу. Иначе прогон 20 вопросов —
    это много минут полностью пустого окна статуса, где невозможно отличить
    «агент думает» от «всё зависло».
    """
    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID")  # Get the SPACE_ID for sending link to the code

    if profile:
        username = f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        yield "Please Login to Hugging Face with the button.", None
        return

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Agent
    yield "Инициализация агента...", None
    try:
        agent = GaiaAgent(api_url=api_url)
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        yield f"Error initializing agent: {e}", None
        return

    # In the case of an app running as a Hugging Face space, this link points
    # toward your codebase (useful for others, so please keep it public)
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    yield "Получение списка вопросов...", None
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            print("Fetched questions list is empty.")
            yield "Fetched questions list is empty or invalid format.", None
            return
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.JSONDecodeError as e:
        # ВАЖНО: JSONDecodeError — подкласс RequestException, поэтому этот
        # except должен идти РАНЬШЕ общего, иначе он недостижим.
        print(f"Error decoding JSON response from questions endpoint: {e}")
        print(f"Response text: {response.text[:500]}")
        yield f"Error decoding server response for questions: {e}", None
        return
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        yield f"Error fetching questions: {e}", None
        return
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        yield f"An unexpected error occurred fetching questions: {e}", None
        return

    # 3. Run your Agent
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")
    total = len(questions_data)
    for index, item in enumerate(questions_data, start=1):
        task_id = item.get("task_id")
        question_text = item.get("question")
        file_name = item.get("file_name")  # может быть пустой строкой или отсутствовать
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue

        # Показываем, за какой вопрос агент взялся, ДО его запуска: один вопрос
        # может думать минуту с лишним, и без этого непонятно, где он застрял.
        yield (
            f"Вопрос {index} из {total}...\n"
            f"{question_text[:150]}{'...' if len(question_text) > 150 else ''}",
            pd.DataFrame(results_log) if results_log else None,
        )

        try:
            submitted_answer = agent(question_text, task_id=task_id, file_name=file_name)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append(
                {"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer}
            )
        except Exception as e:
            print(f"Error running agent on task {task_id}: {e}")
            results_log.append(
                {"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"}
            )

        # Отдаём уже полученные ответы в таблицу — их видно по ходу прогона,
        # а не только в самом конце.
        yield (
            f"Готово {index} из {total}. Ответов собрано: {len(answers_payload)}.",
            pd.DataFrame(results_log),
        )

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        yield "Agent did not produce any answers to submit.", pd.DataFrame(results_log)
        return

    # 4. Prepare Submission
    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    # 5. Submit
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(results_log)
        yield final_status, results_df
        return
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        yield status_message, results_df
        return
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        results_df = pd.DataFrame(results_log)
        yield status_message, results_df
        return
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        yield status_message, results_df
        return
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        yield status_message, results_df
        return


# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# GAIA Agent Evaluation Runner")
    gr.Markdown(
        """
        **Инструкция:**
        1. Войдите в свой аккаунт Hugging Face кнопкой ниже (нужно для отправки результата под вашим именем).
        2. Нажмите «Run Evaluation & Submit All Answers», чтобы агент прошёл все вопросы и отправил ответы.

        ---
        Агент — `smolagents.CodeAgent` с шестью инструментами: веб-поиск, чтение веб-страниц,
        скачивание приложенных к вопросам файлов, распознавание речи в аудио, расшифровка
        видео YouTube и анализ изображений (см. `agent.py` и `tools.py`).

        Прогон всех вопросов занимает несколько минут. Прогресс виден в поле статуса,
        а ответы появляются в таблице по мере готовности — ждать до самого конца не нужно.
        """
    )

    gr.LoginButton()
    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(fn=run_and_submit_all, outputs=[status_output, results_table])

if __name__ == "__main__":
    print("\n" + "-" * 30 + " App Starting " + "-" * 30)
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID")

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️ SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup:
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️ SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-" * (60 + len(" App Starting ")) + "\n")
    print("Launching Gradio Interface for GAIA Agent Evaluation...")
    demo.launch(debug=True, share=False)
