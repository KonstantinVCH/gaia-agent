---
title: GAIA Agent (Unit 4 Final Assignment)
emoji: 🕵️
colorFrom: yellow
colorTo: blue
sdk: gradio
sdk_version: 6.25.0
app_file: app.py
pinned: false
hf_oauth: true
hf_oauth_expiration_minutes: 480
---

# GAIA Agent — Hugging Face Agents Course, Unit 4

A `smolagents.CodeAgent` that answers questions from the course subset of the
[GAIA benchmark](https://huggingface.co/spaces/gaia-benchmark/leaderboard) and submits
its answers to the course scoring API.

## Tools

| Tool | Purpose |
|---|---|
| `web_search` (DuckDuckGo) | look up facts on the web |
| `visit_webpage` | read the pages it finds |
| `download_gaia_file` | fetch the file attached to a question (`GET /files/{task_id}`) |
| `transcribe_audio` | speech-to-text for attached `.mp3`/`.wav` (Whisper via HF Inference API) |
| `get_youtube_transcript` | subtitles of a YouTube video referenced in a question |
| `describe_image` | vision: analyse an attached image via a VLM |
| `read_pdf` | extract text from a PDF by URL or path, with substring search |
| `read_text_file` | read a local .py/.txt/.json file (`open` is banned in the sandbox) |
| `wikipedia` | Wikipedia API directly; `mode='tables'` returns article tables for counting |

### Inference providers

Set `GROQ_API_KEY` to run the model through Groq (free, no card) instead of the Hugging Face
Inference API — useful when the HF monthly quota is exhausted. Groq covers all three needs:
chat (`openai/gpt-oss-120b`), speech-to-text (`whisper-large-v3`) and vision (`qwen/qwen3.6-27b`).
`HF_TOKEN` is still required for Space creation and user login. When both are set Groq wins,
with automatic fallback to HF if a Groq call fails.

Plus the built-in sandboxed Python interpreter, with `pandas`/`openpyxl` explicitly
authorized so the agent can parse attached spreadsheets in code it writes itself.

The tool set was chosen against the actual question set (fetched from the public
`GET /questions` endpoint), which contains questions with `.mp3`, `.xlsx` and `.py`
attachments, image attachments, and YouTube links.

## Configuration

Set these as **Space secrets** (Settings → Variables and secrets):

| Secret | Required | Default |
|---|---|---|
| `HF_TOKEN` | yes | — |
| `MODEL_ID` | no | `Qwen/Qwen2.5-Coder-32B-Instruct` |
| `VISION_MODEL_ID` | no | `Qwen/Qwen2.5-VL-72B-Instruct` |

## Why `sdk_version` is pinned to 6.25.0

`app.py` derives from the official course template, which targeted Gradio 5. Gradio 6 has
since shipped, so this was worth checking rather than assuming: Gradio 6 still supports the
one thing the app depends on — auto-injection of `gr.OAuthProfile | None` into the event
handler (see `special_args` in `gradio/helpers.py`) — and `gr.LoginButton`, `gr.DataFrame`
and the rest of the UI build cleanly on 6.25.0. `test_app_flow.py` exercises the whole
run/submit flow against that version.

So the pin names the version the code was actually verified against, rather than freezing it
on an aging Gradio 5 that Spaces may eventually stop offering.

## Notes

Answers are normalized before submission (`normalize_answer` in `agent.py`), because the
scoring API compares strings exactly — a stray trailing period or a `FINAL ANSWER:`
prefix turns a semantically correct answer into a wrong one. Run `python test_normalize.py`, `python test_wikipedia.py`, `python test_model_select.py`, `python test_audit_fixes.py`, `python test_space_startup.py` and `python test_app_flow.py` to exercise
the logic offline, and `python test_local.py` to try the agent against real questions without submitting.
