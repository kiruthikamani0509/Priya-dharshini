import os
import time
from dotenv import load_dotenv
from google import genai

load_dotenv()
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
FALLBACK = os.getenv("GEMINI_FALLBACK_MODEL", "")
_client = None


def _call(model: str, prompt: str) -> str:
    resp = _client.models.generate_content(model=model, contents=prompt)
    return resp.text or "Error: empty response from model."


def ask_gemini(prompt: str) -> str:
    global _client
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        return "Error: GEMINI_API_KEY not set. Create a .env file (see .env.example)."
    if _client is None:
        _client = genai.Client(api_key=key)

    models = [MODEL] + ([FALLBACK] if FALLBACK else [])
    last_error = ""
    for model in models:
        for attempt in range(4):
            try:
                return _call(model, prompt)
            except Exception as e:
                last_error = str(e)
                if "503" in last_error or "429" in last_error:
                    time.sleep(2 * (attempt + 1))  # 2s, 4s, 6s, 8s
                    continue
                break
    return f"Error: {last_error}"