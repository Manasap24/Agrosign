# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.staticfiles import StaticFiles
# from pydantic import BaseModel

# from manual_translator import translate_manual

# app = FastAPI(title="AgroSign API")

# # Allow React frontend
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:5173"],  # Vite React
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Serve videos
# app.mount("/videos", StaticFiles(directory="../sign_videos"), name="videos")


# class TextRequest(BaseModel):
#     text: str
#     language: str = "english"


# @app.get("/")
# def home():
#     return {"message": "AgroSign Backend Running"}


# # @app.post("/translate")
# # def translate(request: TextRequest):
# #     result = translate_manual(request.text)

# #     # Convert local file paths to URLs
# #     video_urls = []

# #     for path in result["complete_video_sequence"]:
# #         filename = path.split("\\")[-1].split("/")[-1]
# #         video_urls.append(f"http://127.0.0.1:8000/videos/{filename}")

# #     result["complete_video_sequence"] = video_urls

# #     return result

# @app.post("/translate")
# def translate(request: TextRequest):
#     result = translate_manual(request.text)

#     # Convert local file paths to URLs
#     video_urls = []

#     for path in result["complete_video_sequence"]:
#         filename = path.split("\\")[-1].split("/")[-1]
#         video_urls.append(
#             f"http://127.0.0.1:8000/videos/{filename}"
#         )

#     # Get ALL detected processes from translations
#     process_list = [
#         item["process_name"]
#         for item in result["translations"]
#         if item.get("process_name")
#     ]

#     # Update response
#     result["complete_video_sequence"] = video_urls
#     result["process_sequence"] = process_list

#     return result


# from fastapi import UploadFile, File, Form, HTTPException
# from pathlib import Path
# import shutil
# import tempfile

# from speech_to_text.vosk_hindi import transcribe_hindi
# from speech_to_text.faster_whisper_english import transcribe_english
# from speech_to_text.translate_hindi_to_english import translate_hindi_to_english


# @app.post("/speech-to-text")
# async def speech_to_text(
#     file: UploadFile = File(...),
#     language: str = Form(...)
# ):
#     temp_path = None

#     try:
#         suffix = Path(file.filename).suffix

#         with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#             shutil.copyfileobj(file.file, tmp)
#             temp_path = tmp.name

#         # Hindi -> transcript in Hindi + translation in English
#         if language.lower() == "hindi":
#             hindi_text = transcribe_hindi(temp_path)
#             english_text = translate_hindi_to_english(hindi_text)

#             return {
#                 "language": "hindi",
#                 "transcript": hindi_text,
#                 "translation": english_text
#             }

#         # English -> just the English transcript, no translation needed
#         elif language.lower() == "english":
#             english_text = transcribe_english(temp_path)

#             return {
#                 "language": "english",
#                 "transcript": english_text,
#                 "translation": None
#             }

#         elif language.lower() == "kannada":
#             raise HTTPException(status_code=400, detail="Kannada model not implemented yet")

#         else:
#             raise HTTPException(status_code=400, detail="Unsupported language")

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

#     finally:
#         if temp_path:
#             Path(temp_path).unlink(missing_ok=True)




import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from manual_translator import translate_manual
from speech_to_text.vosk_hindi import transcribe_hindi
from speech_to_text.faster_whisper_english import transcribe_english
from speech_to_text.translate_hindi_to_english import translate_hindi_to_english
from speech_to_text.vosk_hindi_live import create_recognizer, SAMPLE_RATE
from speech_to_text.english_live import (
    transcribe_audio,
    SAMPLE_RATE as WHISPER_SAMPLE_RATE,
)


# ---------------------------------------------------------------------------
# App setup — ONE FastAPI instance, everything registers on this.
# ---------------------------------------------------------------------------
app = FastAPI(title="AgroSign API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve videos
app.mount("/videos", StaticFiles(directory="../sign_videos"), name="videos")

CHUNK_SECONDS = 3  # how much audio to buffer before running Whisper on English


class TextRequest(BaseModel):
    text: str
    language: str = "english"


@app.get("/")
def home():
    return {"message": "AgroSign Backend Running"}


# ---------------------------------------------------------------------------
# TEXT -> SIGN
# ---------------------------------------------------------------------------
@app.post("/translate")
def translate(request: TextRequest):
    result = translate_manual(request.text)

    # Convert local file paths to URLs
    video_urls = []

    for path in result["complete_video_sequence"]:
        filename = path.split("\\")[-1].split("/")[-1]
        video_urls.append(
            f"http://127.0.0.1:8000/videos/{filename}"
        )

    # Get ALL detected processes from translations
    process_list = [
        item["process_name"]
        for item in result["translations"]
        if item.get("process_name")
    ]

    result["complete_video_sequence"] = video_urls
    result["process_sequence"] = process_list

    return result


# ---------------------------------------------------------------------------
# FILE UPLOAD -> SPEECH TO TEXT (non-live, existing feature)
# ---------------------------------------------------------------------------
@app.post("/speech-to-text")
async def speech_to_text(
    file: UploadFile = File(...),
    language: str = Form(...)
):
    temp_path = None

    try:
        suffix = Path(file.filename).suffix

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name

        # Hindi -> transcript in Hindi + translation in English
        if language.lower() == "hindi":
            hindi_text = transcribe_hindi(temp_path)
            english_text = translate_hindi_to_english(hindi_text)

            return {
                "language": "hindi",
                "transcript": hindi_text,
                "translation": english_text
            }

        # English -> just the English transcript, no translation needed
        elif language.lower() == "english":
            english_text = transcribe_english(temp_path)

            return {
                "language": "english",
                "transcript": english_text,
                "translation": None
            }

        elif language.lower() == "kannada":
            raise HTTPException(status_code=400, detail="Kannada model not implemented yet")

        else:
            raise HTTPException(status_code=400, detail="Unsupported language")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_path:
            Path(temp_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# LIVE HINDI  (Vosk streaming ASR + Hindi -> English translation)
# ---------------------------------------------------------------------------
@app.websocket("/live-hindi")
async def live_hindi(websocket: WebSocket):

    await websocket.accept()

    print("React connected to live Hindi")

    try:

        recognizer = create_recognizer()

        await websocket.send_json({
            "type": "connected",
            "message": "Live Hindi speech recognition started"
        })

        while True:

            audio_data = await websocket.receive_bytes()

            if not audio_data:
                continue

            if recognizer.AcceptWaveform(audio_data):

                result = json.loads(
                    recognizer.Result()
                )

                hindi_text = result.get(
                    "text",
                    ""
                ).strip()

                if hindi_text:

                    print("Hindi:", hindi_text)

                    try:

                        english_text = translate_hindi_to_english(
                            hindi_text
                        )

                    except Exception as e:

                        print(
                            "Translation error:",
                            e
                        )

                        english_text = ""

                    print(
                        "English:",
                        english_text
                    )

                    await websocket.send_json({
                        "type": "final",
                        "hindi": hindi_text,
                        "english": english_text
                    })

            else:

                result = json.loads(
                    recognizer.PartialResult()
                )

                partial_text = result.get(
                    "partial",
                    ""
                ).strip()

                if partial_text:

                    await websocket.send_json({
                        "type": "partial",
                        "hindi": partial_text
                    })

    except WebSocketDisconnect:

        print(
            "React disconnected from live Hindi"
        )

    except Exception as e:

        print(
            "Live Hindi error:",
            e
        )

        try:

            await websocket.send_json({
                "type": "error",
                "error": str(e)
            })

        except Exception:
            pass


# ---------------------------------------------------------------------------
# LIVE ENGLISH  (faster-whisper, buffered streaming ASR)
# ---------------------------------------------------------------------------
@app.websocket("/live-english")
async def live_english(websocket: WebSocket):

    await websocket.accept()

    print("React connected to live English")

    buffer = np.empty((0,), dtype=np.float32)

    try:

        await websocket.send_json({
            "type": "connected",
            "message": "Live English speech recognition started"
        })

        while True:

            audio_data = await websocket.receive_bytes()

            if not audio_data:
                continue

            # Incoming bytes are 16-bit PCM (little-endian) at 16kHz,
            # matching what the frontend downsamples to for /live-hindi too.
            pcm16 = np.frombuffer(audio_data, dtype=np.int16)
            audio_f32 = pcm16.astype(np.float32) / 32768.0

            buffer = np.concatenate((buffer, audio_f32))

            if len(buffer) >= WHISPER_SAMPLE_RATE * CHUNK_SECONDS:

                text = transcribe_audio(buffer)

                if text:

                    print("English:", text)

                    await websocket.send_json({
                        "type": "final",
                        "english": text
                    })

                buffer = np.empty((0,), dtype=np.float32)

    except WebSocketDisconnect:

        print(
            "React disconnected from live English"
        )

    except Exception as e:

        print(
            "Live English error:",
            e
        )

        try:

            await websocket.send_json({
                "type": "error",
                "error": str(e)
            })

        except Exception:
            pass