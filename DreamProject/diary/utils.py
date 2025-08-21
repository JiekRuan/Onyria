import os
import json
import math
import tempfile
import logging
from datetime import datetime
from django.core.files.base import ContentFile
from django.db.models import Count
from django.db.models.functions import TruncDate
from dotenv import load_dotenv
from django.conf import settings
from groq import Groq
from mistralai import Mistral
from collections import Counter
from .models import Dream

# Configuration du logging professionnel
logger = logging.getLogger(__name__)

# Chargement des variables d'environnement
load_dotenv()

# Constantes de configuration
WHISPER_MODEL = "whisper-large-v3-turbo"
DEFAULT_TEMPERATURE = 0.0
MAX_FALLBACK_RETRIES = 3
IMAGE_GENERATION_MODEL = "mistral-medium-2505"

BASE_DIR = settings.BASE_DIR
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")

# Clients externes
groq_client = Groq(api_key=GROQ_API_KEY)
mistral_client = Mistral(api_key=MISTRAL_API_KEY)

# ---------- UTILS ----------

def read_file(file_path):
    """Lit un fichier depuis /prompt avec encodage UTF-8"""
    path = os.path.join(BASE_DIR, "diary", "prompt", file_path)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def softmax(preds):
    """Applique softmax à un dictionnaire de prédictions"""
    exp = {k: math.exp(v) for k, v in preds.items()}
    total = sum(exp.values())
    return {k: v / total for k, v in exp.items()}

def validate_and_fix_interpretation(interpretation_data):
    """
    Valide et corrige le format de l'interprétation si nécessaire
    Garantit un format cohérent avec des valeurs string
    """
    if interpretation_data is None:
        logger.warning("Interprétation None reçue")
        return None
    
    expected_keys = ["Émotionnelle", "Symbolique", "Cognitivo-scientifique", "Freudien"]
    fixed_interpretation = {}
    
    logger.info(f"Validation interprétation - Type reçu: {type(interpretation_data)}")
    
    for key in expected_keys:
        if key in interpretation_data:
            value = interpretation_data[key]
            
            # Si c'est un objet avec 'contenu', extraire le contenu
            if isinstance(value, dict) and 'contenu' in value:
                fixed_interpretation[key] = value['contenu']
                logger.debug(f"Extraction contenu pour {key}")
            # Si c'est un objet avec 'content', extraire le content  
            elif isinstance(value, dict) and 'content' in value:
                fixed_interpretation[key] = value['content']
                logger.debug(f"Extraction content pour {key}")
            # Si c'est déjà une string, la garder
            elif isinstance(value, str):
                fixed_interpretation[key] = value
                logger.debug(f"String directe pour {key}")
            # Sinon, convertir en string
            else:
                fixed_interpretation[key] = str(value)
                logger.warning(f"Conversion forcée en string pour {key}: {type(value)}")
        else:
            # Clé manquante, ajouter un placeholder
            fixed_interpretation[key] = "Interprétation non disponible"
            logger.warning(f"Clé manquante: {key}")
    
    logger.info("Validation interprétation terminée avec succès")
    return fixed_interpretation

# ---------- TRANSCRIPTION ----------

def transcribe_audio(audio_data, language="fr"):
    """Transcrit un audio en texte avec Whisper de Groq"""
    logger.info(f"Début transcription audio - Langue: {language}")
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name

        with open(temp_file_path, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                file=audio_file,
                model=WHISPER_MODEL,
                prompt="Specify context or spelling",
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"],
                language=language,
                temperature=DEFAULT_TEMPERATURE,
            )

        os.unlink(temp_file_path)
        logger.info(f"Transcription réussie - {len(transcription.text)} caractères")
        return transcription.text

    except Exception as e:
        logger.error(f"Échec transcription audio: {e}")
        return None

# ---------- FALLBACK SYSTEM ----------

def safe_mistral_call(model, messages, operation="API call"):
    """
    Appel Mistral sécurisé avec système de fallback automatique
    
    Args:
        model: Modèle principal à utiliser
        messages: Messages pour l'API
        operation: Description de l'opération (pour les logs)
    
    Returns:
        Response de l'API ou None si tous les fallbacks échouent
    """
    logger.info(f"[{operation}] Démarrage avec modèle: {model}")
    
    # Hiérarchie de fallback par modèle
    fallback_chain = {
        "mistral-large-latest": ["mistral-medium", "mistral-small-latest", "open-mistral-7b"],
        "mistral-medium": ["mistral-small-latest", "open-mistral-7b"],
        "mistral-small-latest": ["open-mistral-7b"],
        "open-mistral-7b": []
    }
    
    models_to_try = [model] + fallback_chain.get(model, [])
    
    for attempt, current_model in enumerate(models_to_try):
        try:
            logger.info(f"[{operation}] Tentative {attempt + 1}: {current_model}")
            
            response = mistral_client.chat.complete(
                model=current_model,
                messages=messages,
                response_format={"type": "json_object"},
            )
            
            if attempt > 0:
                logger.warning(f"[{operation}] Fallback réussi avec {current_model}")
            
            return response
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Erreurs qui nécessitent un fallback
            if any(keyword in error_msg for keyword in [
                "insufficient_quota", "quota_exceeded", "rate_limit", 
                "model_not_found", "service_unavailable", "timeout"
            ]):
                logger.warning(f"[{operation}] Erreur {current_model}: {e}")
                
                if attempt == len(models_to_try) - 1:
                    logger.error(f"[{operation}] Tous les fallbacks échoués")
                    return None
                    
                continue
            else:
                logger.error(f"[{operation}] Erreur critique {current_model}: {e}")
                raise e
    
    return None

# ---------- ANALYSE D'ÉMOTIONS ----------

def analyze_emotions(text):
    """Renvoie le score des émotions + l'émotion dominante avec fallback"""
    logger.info("Début analyse émotionnelle")
    
    system_prompt = read_file("context_emotion.txt")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]
    
    response = safe_mistral_call(
        model="mistral-small-latest",
        messages=messages,
        operation="Analyse émotionnelle"
    )
    
    if response is None:
        logger.error("Échec analyse émotionnelle - tous les modèles indisponibles")
        return None, None
    
    try:
        scores = softmax(json.loads(response.choices[0].message.content))
        dominant = max(scores.items(), key=lambda x: x[1])
        logger.info(f"Émotion dominante détectée: {dominant[0]} ({dominant[1]:.2f})")
        return scores, dominant
        
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Erreur parsing émotions: {e}")
        return None, None

def classify_dream(emotions):
    """Détermine si le rêve est un cauchemar ou non"""
    logger.info("Classification du type de rêve")
    
    if emotions is None:
        logger.warning("Classification impossible - émotions non disponibles")
        return None
    
    with open(os.path.join(BASE_DIR, "diary", "prompt", "reference_emotions.json")) as f:
        ref = json.load(f)
    
    pos = [emotions[e] for e in ref["positif"] if e in emotions]
    neg = [emotions[e] for e in ref["negatif"] if e in emotions]
    
    avg_pos = sum(pos) / len(pos or [1])
    avg_neg = sum(neg) / len(neg or [1])
    
    classification = "cauchemar" if avg_neg > avg_pos else "rêve"
    logger.info(f"Classification: {classification}")
    
    return classification

# ---------- INTERPRÉTATION ----------

def interpret_dream(text):
    """Demande à Mistral une interprétation du rêve avec fallback et validation"""
    logger.info("Début interprétation du rêve")
    
    system_prompt = read_file("context_interpretation.txt")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]
    
    response = safe_mistral_call(
        model="mistral-large-latest",
        messages=messages,
        operation="Interprétation de rêve"
    )
    
    if response is None:
        logger.error("Échec interprétation - tous les modèles indisponibles")
        return None
    
    try:
        raw_interpretation = json.loads(response.choices[0].message.content)
        logger.info("Réponse IA reçue, validation en cours...")
        
        # Valider et corriger le format
        validated_interpretation = validate_and_fix_interpretation(raw_interpretation)
        
        if validated_interpretation:
            logger.info("Interprétation générée et validée avec succès")
            return validated_interpretation
        else:
            logger.error("Échec validation interprétation")
            return None
            
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Erreur parsing interprétation: {e}")
        return None

# ---------- IMAGE ----------

def generate_image_from_text(user, prompt_text, dream_instance):
    """
    Génère une image IA à partir du texte du rêve, via agent Mistral.
    Stocke l'image en base64 dans le modèle Dream.
    """
    logger.info(f"Génération image pour rêve ID: {dream_instance.id}")
    
    try:
        system_instructions = read_file("instructions_image.txt")

        try:
            agent = mistral_client.beta.agents.create(
                model=IMAGE_GENERATION_MODEL,
                name="Dream Image Agent",
                instructions=system_instructions,
                tools=[{"type": "image_generation"}],
                completion_args={"temperature": 0.3, "top_p": 0.95},
            )

            conversation = mistral_client.beta.conversations.start(
                agent_id=agent.id,
                inputs=prompt_text
            )

            file_id = next(
                (item.file_id for output in conversation.outputs if hasattr(output, "content")
                 for item in output.content if hasattr(item, "file_id")),
                None
            )

            if not file_id:
                logger.warning("Aucune image générée par l'agent")
                return False

            image_bytes = mistral_client.files.download(file_id=file_id).read()

            # Stocker en base64 au lieu de fichier
            dream_instance.set_image_from_bytes(image_bytes, format='PNG')
            dream_instance.save()

            logger.info(f"Image stockée en base64 pour rêve {dream_instance.id}")
            return True

        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in [
                "insufficient_quota", "quota_exceeded", "rate_limit"
            ]):
                logger.warning(f"Quota image atteint: {e}")
                return False
            else:
                raise e

    except Exception as e:
        logger.error(f"Erreur génération image: {e}")
        return False
    
# ---------- PROFIL ONYRIQUE ----------

def get_profil_onirique_stats(user):
    """Calcule les statistiques du profil onirique d'un utilisateur"""
    logger.info(f"Calcul statistiques onirique pour utilisateur: {user.id}")
    
    dreams = Dream.objects.filter(user=user)
    total = dreams.count()

    if total == 0:
        logger.info("Aucun rêve enregistré")
        return {
            "statut_reveuse": "silence onirique",
            "pourcentage_reveuse": 0,
            "label_reveuse": "rêves enregistrés",
            "emotion_dominante": "émotion endormie",
            "emotion_dominante_percentage": 0,
        }

    # Statut rêve vs cauchemar
    nb_reves = dreams.filter(dream_type='rêve').count()
    nb_cauchemars = dreams.filter(dream_type='cauchemar').count()

    if nb_reves >= nb_cauchemars:
        statut_reveuse = "âme rêveuse"
        pourcentage = round((nb_reves / total) * 100)
        label = "rêves"
    else:
        statut_reveuse = "en proie aux cauchemars"
        pourcentage = round((nb_cauchemars / total) * 100)
        label = "cauchemars"

    # Émotion dominante
    emotions = dreams.values_list('dominant_emotion', flat=True)
    emotion_counts = Counter(emotions)

    if emotion_counts:
        emotion_dominante, count = emotion_counts.most_common(1)[0]
        emotion_percentage = round((count / total) * 100)
        logger.info(f"Profil calculé: {statut_reveuse}, émotion dominante: {emotion_dominante}")
    else:
        emotion_dominante = "émotion endormie"
        emotion_percentage = 0

    return {
        "statut_reveuse": statut_reveuse,
        "pourcentage_reveuse": pourcentage,
        "label_reveuse": label,
        "emotion_dominante": emotion_dominante,
        "emotion_dominante_percentage": emotion_percentage,
    }
    
# ---------- DASHBOARD PERSONNEL ----------
#----------- Suivi du type de reve --------
def get_dream_type_stats(user):
    """Calcule les statistiques des types de rêves pour les graphiques"""
    logger.info(f"Calcul statistiques des types de rêves pour utilisateur: {user.id}")
    
    dreams = Dream.objects.filter(user=user)
    total = dreams.count()
    
    if total == 0:
        return {
            'percentages': {'rêve': 0, 'cauchemar': 0},
            'counts': {'rêve': 0, 'cauchemar': 0},
            'total': 0
        }
    
    nb_reves = dreams.filter(dream_type='rêve').count()
    nb_cauchemars = dreams.filter(dream_type='cauchemar').count()
    
    return {
        'percentages': {
            'rêve': round((nb_reves / total) * 100, 1),
            'cauchemar': round((nb_cauchemars / total) * 100, 1)
        },
        'counts': {
            'rêve': nb_reves,
            'cauchemar': nb_cauchemars
        },
        'total': total
    }

def get_dream_type_timeline(user):
    """Récupère l'évolution des types de rêves dans le temps"""

    logger.info(f"Calcul timeline types de rêves pour utilisateur: {user.id}")
    
    dreams = Dream.objects.filter(user=user).annotate(
        date_only=TruncDate('created_at')
    ).values('date_only', 'dream_type').annotate(
        count=Count('id')
    ).order_by('date_only')
    
    # Organiser les données par date
    timeline_data = {}
    for dream in dreams:
        date_str = dream['date_only'].strftime('%Y-%m-%d')
        if date_str not in timeline_data:
            timeline_data[date_str] = {'rêve': 0, 'cauchemar': 0}
        timeline_data[date_str][dream['dream_type']] = dream['count']
    
    # Convertir en liste pour le frontend
    timeline_list = []
    for date_str, counts in sorted(timeline_data.items()):
        timeline_list.append({
            'date': date_str,
            'rêve': counts['rêve'],
            'cauchemar': counts['cauchemar']
        })
    
    return timeline_list

def get_emotions_stats(user):
    """Calcule les statistiques des émotions/humeurs pour les graphiques"""
    logger.info(f"Calcul statistiques des émotions pour utilisateur: {user.id}")
    
    dreams = Dream.objects.filter(user=user).exclude(dominant_emotion__isnull=True)
    total = dreams.count()
    
    if total == 0:
        return {
            'percentages': {},
            'counts': {},
            'total': 0
        }
    
    emotion_counts = Counter(dreams.values_list('dominant_emotion', flat=True))
    
    # Calculer les pourcentages
    emotion_percentages = {}
    for emotion, count in emotion_counts.items():
        emotion_percentages[emotion] = round((count / total) * 100, 1)
    
    return {
        'percentages': emotion_percentages,
        'counts': dict(emotion_counts),
        'total': total
    }


def get_emotions_timeline(user):
    """Récupère l'évolution des émotions dominantes dans le temps"""
    
    logger.info(f"Calcul timeline émotions pour utilisateur: {user.id}")
    
    dreams = Dream.objects.filter(user=user).exclude(
        dominant_emotion__isnull=True
    ).annotate(
        date_only=TruncDate('created_at')
    ).values('date_only', 'dominant_emotion').annotate(
        count=Count('id')
    ).order_by('date_only')
    
    # Organiser les données par date
    timeline_data = {}
    all_emotions = set()
    
    for dream in dreams:
        date_str = dream['date_only'].strftime('%Y-%m-%d')
        emotion = dream['dominant_emotion']
        all_emotions.add(emotion)
        
        if date_str not in timeline_data:
            timeline_data[date_str] = {}
        timeline_data[date_str][emotion] = dream['count']
    
    # Convertir en liste pour le frontend
    timeline_list = []
    for date_str in sorted(timeline_data.keys()):
        entry = {'date': date_str}
        for emotion in all_emotions:
            entry[emotion] = timeline_data[date_str].get(emotion, 0)
        timeline_list.append(entry)
    
    return timeline_list, list(all_emotions)