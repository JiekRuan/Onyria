import json
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.contrib.auth.decorators import login_required
from .models import Dream
from collections import Counter
from .utils import (
    transcribe_audio,
    analyze_emotions,
    classify_dream,
    interpret_dream,
    generate_image_from_text,
    get_profil_onirique_stats,
    get_dream_type_stats,
    get_dream_type_timeline,
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
@csrf_exempt
def analyse_from_voice(request):
    if 'audio' in request.FILES:
        try:
            audio_file = request.FILES['audio']
            audio_data = audio_file.read()

            # Étape 1: Transcription
            transcription = transcribe_audio(audio_data)
            if not transcription:
                return dream_analysis_error()

            # Étape 2: Analyse des émotions
            emotions, dominant_emotion = analyze_emotions(transcription)
            if emotions is None or dominant_emotion is None:
                return dream_analysis_error()

            # Étape 3: Classification du rêve
            dream_type = classify_dream(emotions)
            if dream_type is None:
                return dream_analysis_error()

            # Étape 4: Interprétation
            interpretation = interpret_dream(transcription)
            if interpretation is None:
                return dream_analysis_error()

            # Si tout s'est bien passé, créer le rêve
            dream = Dream.objects.create(
                user=request.user,
                transcription=transcription,
                emotions=emotions,
                dominant_emotion=dominant_emotion[0],
                dream_type=dream_type,
                interpretation=interpretation,
                is_analyzed=True,
            )

            # Génération d'image (peut échouer sans arrêter le processus)
            generate_image_from_text(request.user, transcription, dream)

            # Formatage des labels pour la réponse JSON
            formatted_dominant_emotion = EMOTION_LABELS.get(
                dominant_emotion[0], dominant_emotion[0].capitalize()
            )
            formatted_dream_type = DREAM_TYPE_LABELS.get(
                dream_type, dream_type.capitalize()
            )

            return JsonResponse(
                {
                    "success": True,
                    "transcription": transcription,
                    "emotions": emotions,
                    "dominant_emotion": [formatted_dominant_emotion],
                    "dream_type": formatted_dream_type,
                    "interpretation": interpretation,
                    "image_path": dream.image_url,
                }
            )

        except Exception as e:
            return dream_analysis_error()

    return JsonResponse(
        {'success': False, 'error': 'Pas de fichier audio transmis'}
    )


@login_required
def dream_followup(request):
    """Page de suivi des rêves avec statistiques et graphiques"""

    # Récupération des données
    dream_type_stats = get_dream_type_stats(request.user)
    dream_type_timeline = get_dream_type_timeline(request.user)

    context = {
        'dream_type_stats': dream_type_stats,
        'dream_type_timeline': dream_type_timeline,
        'has_data': dream_type_stats['total'] > 0,
    }

    return render(request, 'diary/dream_followup.html', context)
