import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .manual_translator import translate_manual

app = FastAPI(title="AgroSign API")

frontend_url = os.getenv("FRONTEND_URL", "http://localhost:5173")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent

app.mount("/videos", StaticFiles(directory=BASE_DIR / "sign_videos"), name="videos")


class TextRequest(BaseModel):
    text: str
    language: str = "english"


@app.get("/")
def home():
    return {"message": "AgroSign Backend Running"}


@app.post("/translate")
def translate(request: TextRequest, http_request: Request):
    result = translate_manual(request.text)

    video_urls = []

    base_url = str(http_request.base_url).rstrip("/")

    for path in result["complete_video_sequence"]:
        filename = path.split("\\")[-1].split("/")[-1]
        video_urls.append(f"{base_url}/videos/{filename}")

    result["complete_video_sequence"] = video_urls

    return result
