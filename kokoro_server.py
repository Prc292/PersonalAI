
from contextlib import asynccontextmanager
from io import BytesIO

import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from kokoro_mlx import KokoroTTS
import time


tts = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts
    print("Loading Kokoro voice model...")
    tts = KokoroTTS.from_pretrained()
    print("Kokoro is ready.")
    yield
    if tts is not None:
        tts.close()


app = FastAPI(
    title="PersonalAI Local Voice",
    lifespan=lifespan,
)


class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1500)
    voice: str = "am_onyx"
    speed: float = Field(default=1.0, ge=0.7, le=1.3)


@app.get("/health")
def health():
    return {"status": "ready" if tts is not None else "loading"}



@app.post("/speech")
def speech(request: SpeechRequest):
    if tts is None:
        raise HTTPException(status_code=503, detail="Voice model is loading")

    try:
        started = time.perf_counter()

        result = tts.generate(
            request.text,
            voice=request.voice,
            speed=request.speed,
            sample_rate=24000,
        )

        generated = time.perf_counter()
        print(
            f"DEBUG Kokoro generation: "
            f"{generated - started:.2f}s, {len(request.text)} characters"
        )

        buffer = BytesIO()
        sf.write(
            buffer,
            result.audio,
            result.sample_rate,
            format="WAV",
        )

        packaged = time.perf_counter()
        print(f"DEBUG WAV packaging: {packaged - generated:.3f}s")

        return Response(
            content=buffer.getvalue(),
            media_type="audio/wav",
        )

    except Exception as exc:
        print(f"Speech generation error: {exc}")
        raise HTTPException(
            status_code=500,
            detail="Speech generation failed",
        ) from exc
