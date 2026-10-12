import json
import shutil
import tempfile
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import (
    FastAPI,
    UploadFile,
    Request,
    File,
    Form,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from config import VIDEO_DIR
from process_detector import initialize_process_detector
from manual_translator import translate_manual


@asynccontextmanager
async def lifespan(_app):
    # Load the parser/model and precompute static process embeddings once.
    initialize_process_detector()
    yield


app = FastAPI(title="AgroSign API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/videos", StaticFiles(directory=str(VIDEO_DIR)), name="videos")

CHUNK_SECONDS = 3


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
def translate(request: TextRequest, http_request: Request):

    lang = request.language.strip().lower()

    if lang == "hindi":
        from speech_to_text.translate_hindi_to_english import translate_hindi_to_english

        english_text = translate_hindi_to_english(request.text)

        if not english_text:
            raise HTTPException(
                status_code=500,
                detail=f"Translation from {request.language} failed, please try again",
            )

    elif lang == "kannada":
        from speech_to_text.kannada.kannada_to_english import translate_kannada_to_english

        english_text = translate_kannada_to_english(request.text)

        if not english_text:
            raise HTTPException(
                status_code=500,
                detail=f"Translation from {request.language} failed, please try again",
            )

    else:
        english_text = request.text

    result = translate_manual(english_text)

    video_urls = []

    base_url = str(http_request.base_url).rstrip("/")

    for path in result["complete_video_sequence"]:
        filename = path.split("\\")[-1].split("/")[-1]
        video_urls.append(
            f"{base_url}/videos/{filename}"
        )

    process_list = result.get("process_sequence", [])

    def video_url(path):
        filename = Path(path).name
        return f"{base_url}/videos/{filename}"

    result["complete_video_sequence"] = video_urls
    result["process_sequence"] = process_list
    result["translated_input"] = english_text
    # Keep existing response keys and expose event/sign coverage additively.
    for sentence in result.get("translations", []):
        sentence["video_sequence"] = [video_url(path) for path in sentence.get("video_sequence", [])]
        for process in sentence.get("processes", []):
            process["video_sequence"] = [video_url(path) for path in process.get("video_sequence", [])]
            process["renderable_sign_sequence"] = [
                {**{key: value for key, value in clip.items() if key != "path"}, "video_url": video_url(clip["path"])}
                for clip in process.get("renderable_sign_sequence", [])
            ]
        sentence["renderable_sign_sequence"] = [clip for process in sentence.get("processes", []) for clip in process.get("renderable_sign_sequence", [])]
    result["renderable_sign_sequence"] = [
        {**{k: v for k, v in clip.items() if k != "path"}, "video_url": video_url(clip["path"])}
        for clip in result.get("renderable_sign_sequence", [])
    ]
    for fallback in result.get("sign_fallbacks", []):
        paths = fallback.pop("video_paths", [])
        if paths:
            fallback["video_urls"] = [video_url(path) for path in paths]

    return result


# ---------------------------------------------------------------------------
# HINDI / KANNADA TEXT -> ENGLISH TEXT
# ---------------------------------------------------------------------------
class TranslateTextRequest(BaseModel):
    text: str
    language: str


@app.post("/translate-text")
def translate_text_endpoint(request: TranslateTextRequest):

    lang = request.language.strip().lower()

    if lang not in ("hindi", "kannada"):
        raise HTTPException(
            status_code=400,
            detail="Translation is only supported for Hindi and Kannada",
        )

    if not request.text.strip():
        raise HTTPException(
            status_code=400,
            detail="No text provided to translate",
        )

    if lang == "hindi":
        from speech_to_text.translate_hindi_to_english import translate_hindi_to_english

        translated = translate_hindi_to_english(request.text)

    else:
        from speech_to_text.kannada.kannada_to_english import translate_kannada_to_english

        translated = translate_kannada_to_english(request.text)

    if not translated:
        raise HTTPException(
            status_code=500,
            detail="Translation failed, please try again",
        )

    return {
        "translated_text": translated
    }


# ---------------------------------------------------------------------------
# FILE UPLOAD -> SPEECH TO TEXT
# ---------------------------------------------------------------------------
@app.post("/speech-to-text")
async def speech_to_text(
    file: UploadFile = File(...),
    language: str = Form(...)
):

    temp_path = None

    try:

        suffix = Path(file.filename).suffix

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as tmp:

            shutil.copyfileobj(
                file.file,
                tmp
            )

            temp_path = tmp.name

        # Hindi
        if language.lower() == "hindi":
            from speech_to_text.vosk_hindi import transcribe_hindi
            from speech_to_text.translate_hindi_to_english import translate_hindi_to_english

            hindi_text = transcribe_hindi(
                temp_path
            )

            english_text = translate_hindi_to_english(
                hindi_text
            )

            return {
                "language": "hindi",
                "transcript": hindi_text,
                "translation": english_text
            }

        # English
        elif language.lower() == "english":
            from speech_to_text.faster_whisper_english import transcribe_english

            english_text = transcribe_english(
                temp_path
            )

            return {
                "language": "english",
                "transcript": english_text,
                "translation": None
            }

        # Kannada
        elif language.lower() == "kannada":
            from speech_to_text.kannada.kannada_recorded import transcribe_kannada
            from speech_to_text.kannada.kannada_to_english import translate_kannada_to_english

            kannada_text = transcribe_kannada(
                temp_path
            )

            english_text = translate_kannada_to_english(
                kannada_text
            )

            return {
                "language": "kannada",
                "transcript": kannada_text,
                "translation": english_text
            }

        else:

            raise HTTPException(
                status_code=400,
                detail="Unsupported language"
            )

    except HTTPException:
        raise

    except Exception as e:

        # TEMPORARY DEBUG: prints the full traceback to the terminal so
        # we can see exactly which line/error is causing the 500,
        # instead of only the generic message in the HTTP response.
        # Safe to remove once the real issue is found and fixed.
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        if temp_path:
            Path(temp_path).unlink(
                missing_ok=True
            )


# ---------------------------------------------------------------------------
# LIVE HINDI
# ---------------------------------------------------------------------------
@app.websocket("/live-hindi")
async def live_hindi(websocket: WebSocket):
    from speech_to_text.vosk_hindi_live import create_recognizer
    from speech_to_text.translate_hindi_to_english import translate_hindi_to_english

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

                    print(
                        "Hindi:",
                        hindi_text
                    )

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
# LIVE KANNADA
# ---------------------------------------------------------------------------
@app.websocket("/live-kannada")
async def live_kannada(websocket: WebSocket):
    import numpy as np
    from speech_to_text.kannada.kannada_live import transcribe_kannada_audio, SAMPLE_RATE as KANNADA_SAMPLE_RATE
    from speech_to_text.kannada.kannada_to_english import translate_kannada_to_english

    await websocket.accept()

    print("React connected to live Kannada")

    buffer = np.empty(
        (0,),
        dtype=np.float32
    )

    try:

        await websocket.send_json({
            "type": "connected",
            "message": "Live Kannada speech recognition started"
        })

        while True:

            audio_data = await websocket.receive_bytes()

            if not audio_data:
                continue

            pcm16 = np.frombuffer(
                audio_data,
                dtype=np.int16
            )

            audio_f32 = (
                pcm16.astype(np.float32)
                / 32768.0
            )

            buffer = np.concatenate(
                (
                    buffer,
                    audio_f32
                )
            )

            if len(buffer) >= (
                KANNADA_SAMPLE_RATE
                * CHUNK_SECONDS
            ):

                kannada_text = transcribe_kannada_audio(
                    buffer
                )

                if kannada_text:

                    print(
                        "Kannada:",
                        kannada_text
                    )

                    try:

                        english_text = translate_kannada_to_english(
                            kannada_text
                        )

                    except Exception as e:

                        print(
                            "Kannada translation error:",
                            e
                        )

                        english_text = ""

                    print(
                        "English:",
                        english_text
                    )

                    await websocket.send_json({
                        "type": "final",
                        "kannada": kannada_text,
                        "english": english_text
                    })

                buffer = np.empty(
                    (0,),
                    dtype=np.float32
                )

    except WebSocketDisconnect:

        print(
            "React disconnected from live Kannada"
        )

    except Exception as e:

        print(
            "Live Kannada error:",
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
# LIVE ENGLISH
# ---------------------------------------------------------------------------
@app.websocket("/live-english")
async def live_english(websocket: WebSocket):
    import numpy as np
    from speech_to_text.english_live import transcribe_audio, SAMPLE_RATE as WHISPER_SAMPLE_RATE

    await websocket.accept()

    print("React connected to live English")

    buffer = np.empty(
        (0,),
        dtype=np.float32
    )

    try:

        await websocket.send_json({
            "type": "connected",
            "message": "Live English speech recognition started"
        })

        while True:

            audio_data = await websocket.receive_bytes()

            if not audio_data:
                continue

            pcm16 = np.frombuffer(
                audio_data,
                dtype=np.int16
            )

            audio_f32 = (
                pcm16.astype(np.float32)
                / 32768.0
            )

            buffer = np.concatenate(
                (
                    buffer,
                    audio_f32
                )
            )

            if len(buffer) >= (
                WHISPER_SAMPLE_RATE
                * CHUNK_SECONDS
            ):

                text = transcribe_audio(
                    buffer
                )

                if text:

                    print(
                        "English:",
                        text
                    )

                    await websocket.send_json({
                        "type": "final",
                        "english": text
                    })

                buffer = np.empty(
                    (0,),
                    dtype=np.float32
                )

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
