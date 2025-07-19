import json
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .utils import (
    transcribe_audio,
    analyze_emotions,
    classify_dream,
    interpret_dream,
    generate_image_from_text,
)

# ----- Vues principales ----- #

def dream_recorder_view(request):
    """Page d’enregistrement vocal du rêve"""
    return render(request, 'diary/dream_recorder.html')

@csrf_exempt
def transcribe(request):
    """API : reçoit audio et renvoie texte brut"""
    if request.method == 'POST' and 'audio' in request.FILES:
        try:
            audio_file = request.FILES['audio']
            audio_data = audio_file.read()
            transcription = transcribe_audio(audio_data)
            if transcription:
                return JsonResponse({'success': True, 'transcription': transcription})
            else:
                return JsonResponse({'success': False, 'error': 'Échec de la transcription'})
        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})
    return JsonResponse({'success': False, 'error': 'Pas de fichier audio'})

@csrf_exempt
def analyse_from_voice(request):
    """Analyse complète du rêve à partir de l’audio : transcription, émotion, type, interprétation, image"""
    if request.method == 'POST' and 'audio' in request.FILES:
        try:
            audio_file = request.FILES['audio']
            audio_data = audio_file.read()
            transcription = transcribe_audio(audio_data)

            if not transcription:
                return JsonResponse({'success': False, 'error': 'Échec de la transcription'})

            emotions, dominant_emotion = analyze_emotions(transcription)
            dream_type = classify_dream(emotions)
            interpretation = interpret_dream(transcription)
            image_path = generate_image_from_text(transcription)

            return JsonResponse({
                "success": True,
                "transcription": transcription,
                "emotions": emotions,
                "dominant_emotion": dominant_emotion,
                "dream_type": dream_type,
                "interpretation": interpretation,
                "image_path": image_path,
            })

        except Exception as e:
            return JsonResponse({'success': False, 'error': str(e)})

    return JsonResponse({'success': False, 'error': 'Pas de fichier audio transmis'})

def placeholder(request):
    """Vue temporaire pour /diary"""
    return render(request, 'diary/placeholder.html')
