import os
import json
import math
import tempfile
import logging
import time  # nouveau: backoff
from datetime import datetime
from django.core.files.base import ContentFile
from dotenv import load_dotenv
from django.conf import settings
from groq import Groq
from mistralai import Mistral
from collections import Counter
from .models import Dream
import httpx  # nouveau: fallback HTTP direct

# Configuration du logging professionnel
logger = logging.getLogger(__name__)

# Chargement des variables d'environnement
load_dotenv()

# Constantes de configuration
WHISPER_MODEL = "whisper-large-v3-turbo"
DEFAULT_TEMPERATURE = 0.0
MAX_FALLBACK_RETRIES = 3
IMAGE_GENERATION_MODEL = "mistral-medium-2505"

# nouveau: paramètres de retry pour la transcription
TRANSCRIBE_MAX_RETRIES = 3
TRANSCRIBE_BACKOFF_BASE = 1.5  # secondes (exponentiel: 1.5, 2.25, 3.38...)

BASE_DIR = settings.BASE_DIR
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")

# Clients externes
groq_client = Groq(
    api_key=GROQ_API_KEY,
    http_client=httpx.Client(http2=False, timeout=30)  # HTTP/1.1 + timeout ↑
)
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

def _is_retryable_transcription_error(err: Exception) -> bool:
    """nouveau: détecte les erreurs réseau/temporaires qui méritent un retry"""
    msg = str(err).lower()
    keywords = [
        "connection error", "connection reset", "connection aborted",
        "timeout", "temporarily unavailable", "service unavailable",
        "tls", "ssl", "proxy", "rate limit", "503", "502", "429",
    ]
    return any(k in msg for k in keywords)

def _transcribe_via_httpx(file_path: str, language: str = "fr") -> str | None:
    """
    nouveau: Fallback direct sur l'API Groq (OpenAI-compatible) en HTTP/1.1 via httpx.
    Désactive HTTP/2 pour éviter certains soucis de handshake/proxy.
    """
    if not GROQ_API_KEY:
        logger.error("HTTPX fallback: GROQ_API_KEY manquante")
        return None

    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}

    # Multipart form-data — liste de tuples pour gérer les champs répétés
    data = [
        ("model", WHISPER_MODEL),
        ("prompt", "Specify context or spelling"),
        ("response_format", "json"),
        ("language", language),
        ("temperature", str(DEFAULT_TEMPERATURE)),
        ("timestamp_granularities[]", "word"),
        ("timestamp_granularities[]", "segment"),
    ]

    try:
        with open(file_path, "rb") as f:
            files = {"file": ("audio.wav", f, "audio/wav")}
            timeout = httpx.Timeout(connect=15.0, read=180.0, write=60.0, pool=60.0)
            limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
            with httpx.Client(http2=False, timeout=timeout, limits=limits, trust_env=True) as client:
                r = client.post(url, headers=headers, data=data, files=files)
                r.raise_for_status()
                payload = r.json()
                text = payload.get("text")
                if text:
                    logger.info(f"HTTPX fallback OK - {len(text)} caractères")
                    return text
                logger.error(f"HTTPX fallback: réponse inattendue {payload}")
                return None
    except Exception as e:
        logger.error(f"HTTPX fallback échec: {e}")
        return None

def transcribe_audio(audio_data, language="fr"):
    """Transcrit un audio en texte avec Whisper de Groq"""
    logger.info(f"Début transcription audio - Langue: {language}")

    # garde-fou si la clé est absente/mal configurée en préprod
    if not GROQ_API_KEY:
        logger.error("Échec transcription audio: GROQ_API_KEY manquante")
        return None
    
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name

        last_error = None

        for attempt in range(1, TRANSCRIBE_MAX_RETRIES + 1):
            try:
                logger.info(f"Transcription tentative {attempt}/{TRANSCRIBE_MAX_RETRIES}")
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

                logger.info(f"Transcription réussie - {len(transcription.text)} caractères")
                return transcription.text

            except Exception as e:
                last_error = e
                if _is_retryable_transcription_error(e) and attempt < TRANSCRIBE_MAX_RETRIES:
                    sleep_s = round(TRANSCRIBE_BACKOFF_BASE ** attempt, 2)
                    logger.warning(f"Transcription erreur réseau (retry dans {sleep_s}s): {e}")
                    time.sleep(sleep_s)
                    continue
                else:
                    # ← ICI : remplace l’ancien logger.error(...)
                    logger.error("Transcription error (%s): %s", type(e).__name__, e)
                    break


        # nouveau: dernier recours via HTTPX (HTTP/1.1)
        logger.info("Tentative fallback HTTPX pour la transcription…")
        return _transcribe_via_httpx(temp_file_path, language)

    finally:
        # Nettoyage du fichier temporaire même en cas d'erreur
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Impossible de supprimer le fichier temporaire: {e}")

# ---------- FALLBACK SYSTEM ----------

def safe_mistral_call(model, messages, operation="API call"):
    """
    Appel Mistral sécurisé avec système de fallback automatique
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
