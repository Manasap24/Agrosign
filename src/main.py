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


# @app.post("/translate")
# def translate(request: TextRequest):
#     result = translate_manual(request.text)

#     # Convert local file paths to URLs
#     video_urls = []

#     for path in result["complete_video_sequence"]:
#         filename = path.split("\\")[-1].split("/")[-1]
#         video_urls.append(f"http://127.0.0.1:8000/videos/{filename}")

#     result["complete_video_sequence"] = video_urls

#     return result

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

    # Update response
    result["complete_video_sequence"] = video_urls
    result["process_sequence"] = process_list

    return result
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
#         # Save uploaded file temporarily
#         suffix = Path(file.filename).suffix

#         with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
#             shutil.copyfileobj(file.file, tmp)
#             temp_path = tmp.name

#         # Hindi
#         if language.lower() == "hindi":
#             hindi_text = transcribe_hindi(temp_path)
#             english_text = translate_hindi_to_english(hindi_text)

#             return {
#                 "language": "hindi",
#                 "transcript": hindi_text,
#                 "translation": english_text
#             }

#         # English
#         elif language.lower() == "english":
#             english_text = transcribe_english(temp_path)

#             return {
#                 "language": "english",
#                 "transcript": english_text,
#                 "translation": english_text
#             }

#         # Kannada placeholder
#         elif language.lower() == "kannada":
#             raise HTTPException(status_code=400, detail="Kannada model not implemented yet")

#         else:
#             raise HTTPException(status_code=400, detail="Unsupported language")

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

#     finally:
#         if temp_path:
#             Path(temp_path).unlink(missing_ok=True)


from fastapi import UploadFile, File, Form, HTTPException
from pathlib import Path
import shutil
import tempfile

from speech_to_text.vosk_hindi import transcribe_hindi
from speech_to_text.faster_whisper_english import transcribe_english
from speech_to_text.translate_hindi_to_english import translate_hindi_to_english


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
