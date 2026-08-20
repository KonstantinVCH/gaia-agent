"""
Custom tools for the GAIA benchmark agent (HF Agents Course Unit 4).

Built-in smolagents tools (web_search, visit_webpage) are used as-is.
This file adds:
  - DownloadGaiaFileTool   — download a file attached to a GAIA question
  - TranscribeAudioTool    — speech-to-text via HF Inference API (Whisper)
  - YoutubeTranscriptTool  — fetch YouTube subtitles/transcript by URL
"""

import os
import re
import tempfile

import requests
from smolagents import Tool


class DownloadGaiaFileTool(Tool):
    name = "download_gaia_file"
    description = (
        "Downloads a file attached to a GAIA question by its task_id "
        "(image, spreadsheet, audio, text file, etc.). "
        "Returns the local path of the saved file so it can be read or "
        "processed further (e.g. pandas for .xlsx/.csv, transcribe_audio "
        "for .mp3, plain open for .txt/.py)."
    )
    inputs = {
        "task_id": {
            "type": "string",
            "description": "The task_id of the question whose file should be downloaded.",
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
                return f"No file attached to task {task_id} (404)."
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            return f"Failed to download file for {task_id}: {e}"

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
            f"File saved locally at: {local_path} "
            f"(original name: {filename or 'unknown'})."
        )


class TranscribeAudioTool(Tool):
    name = "transcribe_audio"
    description = (
        "Transcribes speech in a local audio file and returns the text. "
        "Accepts a file path (e.g. the path returned by download_gaia_file). "
        "Useful for questions like 'listen to the recording and tell me...'."
    )
    inputs = {
        "file_path": {
            "type": "string",
            "description": "Local path to an audio file (.mp3, .wav, etc.).",
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
            return f"Failed to transcribe {file_path}: {e}"


class YoutubeTranscriptTool(Tool):
    name = "get_youtube_transcript"
    description = (
        "Returns the text transcript (subtitles/auto-captions) of a YouTube "
        "video given its URL or video ID. Useful for questions about what was "
        "SAID in a video. Does NOT provide visual analysis — cannot answer "
        "questions about what is SEEN on screen."
    )
    inputs = {
        "url_or_video_id": {
            "type": "string",
            "description": "YouTube video URL or video ID.",
        }
    }
    output_type = "string"

    @staticmethod
    def _extract_video_id(value: str) -> str:
        match = re.search(r"(?:v=|youtu\.be/|embed/)([A-Za-z0-9_-]{11})", value)
        return match.group(1) if match else value

    def forward(self, url_or_video_id: str) -> str:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi

            video_id = self._extract_video_id(url_or_video_id)
            api = YouTubeTranscriptApi()
            fetched = api.fetch(video_id, languages=["en", "ru"])
            text = " ".join(snippet.text for snippet in fetched)
            return text[:8000]
        except Exception as e:
            return f"Failed to get transcript for {url_or_video_id}: {e}"
