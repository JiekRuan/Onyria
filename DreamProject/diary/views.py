from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .utils import transcribe_audio

def diary(request):
    """Page d'accueil avec enregistreur"""
    return render(request, 'diary/dream_recorder.html')

@csrf_exempt
def transcribe(request):
    """API pour transcrire l'audio"""
    if request.method == 'POST' and 'audio' in request.FILES:
        try:
            audio_file = request.FILES['audio']
            audio_data = audio_file.read()
            
            transcription = transcribe_audio(audio_data)
            
            if transcription:
                return JsonResponse({
                    'success': True,
                    'transcription': transcription
                })
            else:
                return JsonResponse({
                    'success': False,
                    'error': 'Échec de la transcription'
                })
                
        except Exception as e:
            return JsonResponse({
                'success': False,
                'error': str(e)
            })
    
    return JsonResponse({'success': False, 'error': 'Pas de fichier audio'})