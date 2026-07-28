from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from manual_translator import translate_manual

app = FastAPI(title="AgroSign API")

# Allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite React
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve videos
app.mount("/videos", StaticFiles(directory="../sign_videos"), name="videos")


class TextRequest(BaseModel):
    text: str
    language: str = "english"


@app.get("/")
def home():
    return {"message": "AgroSign Backend Running"}


@app.post("/translate")
def translate(request: TextRequest):
    result = translate_manual(request.text)

    # Convert local file paths to URLs
    video_urls = []

    for path in result["complete_video_sequence"]:
        filename = path.split("\\")[-1].split("/")[-1]
        video_urls.append(f"http://127.0.0.1:8000/videos/{filename}")

    result["complete_video_sequence"] = video_urls

    return result