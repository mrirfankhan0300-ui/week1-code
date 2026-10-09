"""CounterWatch AI engine.

Upload a photo of a counter or chair -> Gemini says whether a person is there.

Run:
    export GEMINI_API_KEY="your-key"      # free key from https://aistudio.google.com
    uvicorn main:app --reload
Then open http://127.0.0.1:8000/docs to try it.
"""
import asyncio
import os
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from google import genai
from google.genai import errors, types
from pydantic import BaseModel

load_dotenv()  # reads GEMINI_API_KEY from the .env file

# Model name is configurable in case Google renames/retires models.
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_BYTES = 5 * 1024 * 1024  # 5 MB

PROMPT = (
    "You are checking whether a chair or a service counter is occupied. "
    "First, describe in one short sentence what you see on or at the chair/counter. "
    "Then set person_present to true ONLY if you can clearly see a real, living "
    "human (any visible body part such as face, head, arms, hands or legs) sitting "
    "at or standing at it. "
    "Set person_present to false if it is empty, even when there are bags, clothes, "
    "jackets, objects, mannequins, drawings, posters, photos or reflections of people. "
    "Answer only with the requested JSON."
)


class Detection(BaseModel):
    """What we ask Gemini to return (description first, then the answer)."""
    description: str
    person_present: bool


class Result(BaseModel):
    """What our API returns to the caller."""
    status: Literal["occupied", "empty"]


app = FastAPI(title="CounterWatch AI Engine")
_client = None


def get_client() -> genai.Client:
    """Create the Gemini client lazily so the app can start without a key."""
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise HTTPException(500, "GEMINI_API_KEY is not set on the server.")
        _client = genai.Client(api_key=api_key)
    return _client


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze", response_model=Result)
async def analyze(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, "Upload a JPEG, PNG, or WebP image.")

    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image too large (max 5 MB).")

    client = get_client()
    response = None
    for attempt in range(3):  # Gemini is sometimes busy (503), so try up to 3 times
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=[
                    types.Part.from_bytes(data=data, mime_type=file.content_type),
                    PROMPT,
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=Detection,
                    temperature=0,
                ),
            )
            break
        except errors.ServerError as exc:  # 5xx: Google side, worth retrying
            if attempt == 2:
                raise HTTPException(503, f"Gemini is busy, try again later: {exc}")
            await asyncio.sleep(2 * (attempt + 1))  # wait 2s, then 4s
        except Exception as exc:  # bad key, quota, network, etc.
            raise HTTPException(502, f"Gemini request failed: {exc}")

    result = response.parsed
    if result is None:
        raise HTTPException(502, "Gemini returned an unreadable answer.")
    print(f"[{file.filename}] Gemini saw: {result.description} -> person_present={result.person_present}")
    return Result(status="occupied" if result.person_present else "empty")