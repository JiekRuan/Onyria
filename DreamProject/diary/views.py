# diary/views.py
import json
import os
import logging
import httpx
from django.http import HttpResponse, JsonResponse, HttpRequest
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from diary.models import Dream

logger = logging.getLogger(__name__)


@require_http_methods(["GET"])
def dream_recorder_view(request):
    # Rend simplement la page d’enregistrement (adapté à ton template existant)
    return render(request, "diary/record.html")

@require_http_methods(["GET"])
def dream_diary_view(request):
    # Page racine du module diary -> redirige vers la page 'record'
    return redirect("dream_recorder")

@csrf_exempt
@require_http_methods(["POST"])
def transcribe(request):
    # Alias pour compat ancienne route -> réutilise ton endpoint existant
    return analyse_from_voice(request)

@csrf_exempt
@require_http_methods(["GET"])
def dream_followup(request):
    """
    Endpoint de suivi après l’analyse d’un rêve.
    Remplace le contenu par ta logique (LLM, règles, etc.).
    """
    try:
        payload = json.loads(request.body or "{}")
    except json.JSONDecodeError:
        payload = {}

    question = payload.get("question") or payload.get("message") or ""
    # TODO: branche ta logique réelle ici
    render(request, "diary/followup.html")

def dream_detail_view(request: HttpRequest, dream_id: int) -> HttpResponse:
    dream = get_object_or_404(Dream, id=dream_id)  # or pk=dream_id
    return render(request, "diary/dream_detail.html", {"dream": dream})

# ---- Uniform JSON helpers ----------------------------------------------------
def api_success(text: str):
    """Return a success JSON with multiple aliases so any front key works."""
    return JsonResponse({
        "ok": True,
        "success": True,
        "status": 200,
        "error": None,
        "errorMessage": "",
        "message": text,
        "text": text,
        "transcript": text,
        "result": text,
        "data": {"text": text, "transcript": text, "result": text, "message": text}
    }, status=200)

def api_error(msg: str, http_status: int):
    """Return a uniform error JSON."""
    return JsonResponse({
        "ok": False,
        "success": False,
        "status": http_status,
        "error": msg,
        "errorMessage": msg,
        "message": "",
        "text": "",
        "transcript": "",
        "result": "",
        "data": {"text": "", "transcript": "", "result": "", "message": ""}
    }, status=http_status)

# ---- Accepted audio MIME types and their preferred extensions ----------------
ACCEPTED_MIMES = {
    "audio/flac": ".flac",
    "audio/mpeg": ".mp3",   # mp3/mpga/mpeg
    "audio/mp3": ".mp3",
    "audio/mpga": ".mp3",
    "audio/mp4": ".m4a",    # some browsers use audio/mp4 for m4a
    "video/mp4": ".mp4",    # Safari may report video/mp4
    "audio/ogg": ".ogg",
    "audio/opus": ".opus",
    "audio/wav": ".wav",
    "audio/webm": ".webm",
}

# ---- Main endpoint ------------------------------------------------------------
@csrf_exempt
def analyse_from_voice(request: HttpRequest):
    logger.warning("analyse_from_voice: NEW VIEW ACTIVE")

    # 1) Method guard
    if request.method != "POST":
        return api_error("method_not_allowed", 405)

    # 2) Audio file
    audio = request.FILES.get("audio")
    if not audio:
        return api_error("no_audio", 400)

    # 3) Clean API key (remove CR/LF and spaces)
    api_key = (os.getenv("GROQ_API_KEY") or "").replace("\r", "").replace("\n", "").strip()
    if not api_key:
        logger.error("GROQ_API_KEY missing or invalid")
        return api_error("no_api_key", 500)

    # 4) Groq OpenAI-compatible endpoint
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {"model": "whisper-large-v3"}

    # 5) Normalize filename and MIME for Groq
    orig_name = (audio.name or "audio")
    orig_ctype = (getattr(audio, "content_type", "") or "").split(";", 1)[0].strip().lower()

    mime = orig_ctype if orig_ctype in ACCEPTED_MIMES else ""
    ext = ACCEPTED_MIMES.get(mime)

    if not ext:
        # Try to infer from the original filename extension
        lower = orig_name.lower()
        for mt, ex in ACCEPTED_MIMES.items():
            if lower.endswith(ex):
                mime, ext = mt, ex
                break

    if not mime:
        # Last-resort fallback commonly produced by MediaRecorder
        mime, ext = "audio/webm", ".webm"

    safe_name = orig_name if orig_name.lower().endswith(ext) else f"record{ext}"
    file_bytes = audio.read()
    if not file_bytes:
        return api_error("empty_audio", 400)

    files = {"file": (safe_name, file_bytes, mime)}
    logger.info("Sending to Groq: name=%s, mime=%s, size=%s", safe_name, mime, len(file_bytes))

    # 6) HTTP call with robust error handling (ALWAYS returns a response)
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, headers=headers, data=data, files=files)
            resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        body = e.response.text[:1000] if e.response is not None else ""
        code = e.response.status_code if e.response is not None else 502
        logger.error("Groq HTTPStatusError %s: %s", code, body)
        return api_error(f"groq_http_error: {code} {body}", 502)
    except httpx.HTTPError as e:
        logger.exception("Groq HTTPError (network/timeout/SSL)")
        return api_error(f"transcription_failed: {e}", 502)
    except Exception as e:
        logger.exception("Unexpected server error")
        return api_error(f"server_error: {e}", 500)

    # 7) Success path — parse JSON safely and return unified payload
    try:
        payload = resp.json()
    except Exception:
        return api_error("bad_response_format", 502)

    text = (payload.get("text") or "").strip()
    if not text:
        return api_error("empty_transcription", 502)

    return api_success(text)

# ---- Optional: health check endpoint to validate key/network ------------------
@csrf_exempt
def groq_health(request: HttpRequest):
    """Optional helper: GET /diary/groq_health/ to verify key and outbound network."""
    api_key = (os.getenv("GROQ_API_KEY") or "").replace("\r", "").replace("\n", "").strip()
    if not api_key:
        return api_error("no_api_key", 500)

    url = "https://api.groq.com/openai/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        with httpx.Client(timeout=15.0) as client:
            r = client.get(url, headers=headers)
            r.raise_for_status()
        data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        count = len(data.get("data", [])) if isinstance(data, dict) else None
        return api_success(f"models_count={count}")
    except httpx.HTTPStatusError as e:
        body = e.response.text[:500] if e.response is not None else ""
        code = e.response.status_code if e.response is not None else 502
        return api_error(f"groq_http_error: {code} {body}", 502)
    except httpx.HTTPError as e:
        return api_error(f"network_error: {e}", 502)
    

def dream_diary_view(request):
    # Le plus simple : rediriger vers ta page d’enregistrement existante
    return redirect('/diary/record/')
