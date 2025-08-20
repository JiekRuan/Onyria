import json
import os
from venv import logger
from django.shortcuts import render, get_object_or_404
from django.http import HttpRequest, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
import httpx
from .models import Dream
from collections import Counter
from .utils import (
    transcribe_audio,
    analyze_emotions,
    classify_dream,
    interpret_dream,
    generate_image_from_text,
    get_profil_onirique_stats,
)
from .constants import EMOTION_LABELS, DREAM_TYPE_LABELS, DREAM_ERROR_MESSAGE


def dream_analysis_error():
    """Retourne une réponse JSON d'erreur standardisée"""
    return JsonResponse({'success': False, 'error': DREAM_ERROR_MESSAGE})


# ----- Vues principales ----- #


@login_required
def dream_diary_view(request):
    """Journal des rêves"""
    dreams = Dream.objects.filter(user=request.user).order_by('-created_at')

    stats = get_profil_onirique_stats(request.user)

    # Formatage des labels pour l'affichage
    emotion_dominante = stats.get('emotion_dominante')
    if emotion_dominante:
        stats['emotion_dominante'] = EMOTION_LABELS.get(
            emotion_dominante, emotion_dominante.capitalize()
        )

    statut_reveuse = stats.get('statut_reveuse')
    if statut_reveuse:
        stats['statut_reveuse'] = DREAM_TYPE_LABELS.get(
            statut_reveuse, statut_reveuse.capitalize()
        )

    return render(
        request,
        'diary/dream_diary.html',
        {
            'dreams': dreams,
            **stats,  # déstructure les clés du dict `stats` directement dans le contexte
        },
    )


@login_required
def dream_detail_view(request, dream_id):
    """Affiche les détails d'un rêve spécifique"""
    dream = get_object_or_404(Dream, id=dream_id, user=request.user)

    # Formatage des labels pour l'affichage
    if dream.dominant_emotion:
        formatted_dominant_emotion = EMOTION_LABELS.get(
            dream.dominant_emotion, dream.dominant_emotion.capitalize()
        )
        formatted_dream_type = DREAM_TYPE_LABELS.get(
            dream.dream_type, dream.dream_type.capitalize()
        )
    else:
        formatted_dominant_emotion = "Non analysé"
        formatted_dream_type = "Non analysé"
        
    # Parser l'interprétation si c'est une string JSON
    interpretation = dream.interpretation
    if isinstance(interpretation, str):
        try:
            interpretation = json.loads(interpretation)
        except json.JSONDecodeError:
            interpretation = {}

    context = {
        'dream': dream,
        'formatted_dominant_emotion': formatted_dominant_emotion,
        'formatted_dream_type': formatted_dream_type,
        'interpretation': interpretation,
    }

    return render(request, 'diary/dream_detail.html', context)


@login_required
def dream_recorder_view(request):
    """Page d'enregistrement vocal du rêve"""
    return render(request, 'diary/dream_recorder.html')


@require_http_methods(["POST"])
@csrf_exempt
def transcribe(request):
    """API : reçoit audio et renvoie texte brut"""
    if 'audio' in request.FILES:
        try:
            audio_file = request.FILES['audio']
            audio_data = audio_file.read()
            transcription = transcribe_audio(audio_data)
            if transcription:
                return JsonResponse(
                    {'success': True, 'transcription': transcription}
                )
            else:
                return JsonResponse(
                    {'success': False, 'error': 'Échec de la transcription'}
                )
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    return JsonResponse({'success': False, 'error': 'Pas de fichier audio'})


@require_http_methods(["POST"])
@login_required
@csrf_exempt  # retire si tu utilises déjà un token CSRF côté front
def analyse_from_voice(request: HttpRequest):
    logger.warning("analyse_from_voice: NEW VIEW ACTIVE")

    # 1) Méthode
    if request.method != "POST":
        return JsonResponse({"ok": False, "error": "method_not_allowed"}, status=405)

    # 2) Fichier audio
    audio = request.FILES.get("audio")
    if not audio:
        return JsonResponse({"ok": False, "error": "no_audio"}, status=400)

    # 3) Clé Groq nettoyée
    api_key = (os.getenv("GROQ_API_KEY") or "").replace("\r", "").replace("\n", "").strip()
    if not api_key:
        logger.error("GROQ_API_KEY manquante ou invalide")
        return JsonResponse({"ok": False, "error": "no_api_key"}, status=500)

    # 4) URL/Headers/Data Groq
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {"model": "whisper-large-v3"}

    # 5) PATCH nom/MIME/extension acceptés par Groq
    orig_name = (audio.name or "audio")
    orig_ctype = (audio.content_type or "").split(";", 1)[0].strip().lower()

    ACCEPTED = {
        "audio/flac": ".flac",
        "audio/mpeg": ".mp3",   # mp3/mpga/mpeg
        "audio/mp3": ".mp3",
        "audio/mpga": ".mp3",
        "audio/mp4": ".m4a",    # m4a
        "video/mp4": ".mp4",    # quelques navigateurs
        "audio/ogg": ".ogg",
        "audio/opus": ".opus",
        "audio/wav": ".wav",
        "audio/webm": ".webm",
    }

    mime = orig_ctype if orig_ctype in ACCEPTED else ""
    ext = ACCEPTED.get(mime)

    if not ext:
        lower = orig_name.lower()
        for mt, ex in ACCEPTED.items():
            if lower.endswith(ex):
                mime, ext = mt, ex
                break

    if not mime:
        mime, ext = "audio/webm", ".webm"  # fallback sûr

    safe_name = orig_name if orig_name.lower().endswith(ext) else f"record{ext}"
    file_bytes = audio.read()
    files = {"file": (safe_name, file_bytes, mime)}

    logger.info("Envoi Groq: name=%s, mime=%s, size=%s", safe_name, mime, len(file_bytes))

    # 6) Appel Groq avec gestion d'erreurs QUI RETOURNENT TOUJOURS UNE RÉPONSE
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(url, headers=headers, data=data, files=files)
            resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        body = e.response.text[:1000] if e.response is not None else ""
        code = e.response.status_code if e.response is not None else None
        logger.error("Groq HTTPStatusError %s: %s", code, body)
        return JsonResponse({"ok": False, "error": "groq_http_error", "status": code, "detail": body}, status=502)
    except httpx.HTTPError as e:
        logger.exception("Groq HTTPError (réseau/timeout/SSL)")
        return JsonResponse({"ok": False, "error": "transcription_failed", "detail": str(e)}, status=502)
    except Exception as e:
        logger.exception("Unexpected server error")
        return JsonResponse({"ok": False, "error": "server_error", "detail": str(e)}, status=500)

    # 7) Succès : on renvoie TOUJOURS quelque chose ici
    try:
        payload = resp.json()
    except Exception:
        # Au cas où Groq renverrait du texte brut
        return JsonResponse({"ok": False, "error": "bad_response_format", "detail": resp.text[:500]}, status=502)

    text = payload.get("text") or ""
    return JsonResponse({"ok": True, "text": text}, status=200)



@login_required
def dream_followup(request):
    """Page de suivi des rêves (placeholder)"""
    return render(request, 'diary/dream_followup.html')