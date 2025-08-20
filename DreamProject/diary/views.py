import json
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
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
def analyse_from_voice(request):
    if request.method != "POST":
        return JsonResponse({"ok": False, "error": "method_not_allowed"}, status=405)

    # 1) Récup audio envoyé par le front : FormData.append("audio", file, "record.webm")
    audio = request.FILES.get("audio")
    if not audio:
        return JsonResponse({"ok": False, "error": "no_audio"}, status=400)

    # 2) Clé Groq propre (pas de \n)
    api_key = (os.getenv("GROQ_API_KEY") or "").strip()
    if not api_key or "\n" in api_key or "\r" in api_key:
        return JsonResponse({"ok": False, "error": "invalid_api_key"}, status=500)

    # 3) Appel Groq (OpenAI-compatible)
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {"model": "whisper-large-v3"}  # adapte si besoin
    files = {
        "file": (audio.name, audio.read(), audio.content_type or "audio/webm")
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, headers=headers, data=data, files=files)
            resp.raise_for_status()
        text = resp.json().get("text", "")
        return JsonResponse({"ok": True, "text": text}, status=200)

    except httpx.HTTPStatusError as e:
        # Remonte l’erreur Groq (utile en debug)
        return JsonResponse({
            "ok": False,
            "error": "groq_http_error",
            "status": e.response.status_code,
            "detail": e.response.text[:500]
        }, status=502)
    except httpx.HTTPError:
        return JsonResponse({"ok": False, "error": "transcription_failed"}, status=502)
    except Exception:
        return JsonResponse({"ok": False, "error": "server_error"}, status=500)


@login_required
def dream_followup(request):
    """Page de suivi des rêves (placeholder)"""
    return render(request, 'diary/dream_followup.html')