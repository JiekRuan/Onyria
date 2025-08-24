import os
import json
import math
import time
import tempfile
import logging
import spacy
import re
import nltk
import httpx

from typing import List

from datetime import datetime, timedelta
from django.utils import timezone
from django.core.files.base import ContentFile
from django.db.models import Count
from django.db.models.functions import TruncDate
from dotenv import load_dotenv
from django.conf import settings
from groq import Groq
from mistralai import Mistral
from collections import Counter, defaultdict
from .models import Dream
from .constants import THEME_CATEGORIES

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Chargement des variables d'environnement
load_dotenv()

logger = logging.getLogger(__name__)

try:
    nlp = spacy.load("fr_core_news_sm")
except OSError:
    logger.warning(
        "Modèle spaCy français non trouvé. Installer avec: python -m spacy download fr_core_news_sm"
    )
    nlp = None

try:
    nltk.download('stopwords', quiet=True)
    FRENCH_STOPWORDS = set(stopwords.words('french'))
    stemmer = SnowballStemmer('french')
except ImportError:
    logger.warning("NLTK non disponible. Installer avec: pip install nltk")
    FRENCH_STOPWORDS = set()
    stemmer = None

DREAM_SPECIFIC_STOPWORDS = {
    # Mots du rêve
    'rêve',
    'rêver',
    'dormir',
    'nuit',
    'moment',
    'fois',
    'chose',
    'truc',
    'machin',
    'endroit',
    'côté',
    'genre',
    'espèce',
    'sorte',
    'façon',
    'maniÃ¨re',
    'air',
    'impression',
    'sentiment',
    'sensation',
    'souvenir',
    'souviens',
    'rappelle',
    'crois',
    'pense',
    'imagine',
    'semble',
    # Verbes trop génériques
    'faire',
    'avoir',
    'être',
    'aller',
    'dire',
    'voir',
    'savoir',
    'pouvoir',
    'vouloir',
    'devoir',
    'prendre',
    'donner',
    'mettre',
    'partir',
    'venir',
    'arriver',
    'passer',
    'rester',
    'devenir',
    'porter',
    'regarder',
    'entendre',
    'sentir',
    'trouver',
    'laisser',
    'suivre',
    'montrer',
    'demander',
    'parler',
    'tenir',
    'jouer',
    'tourner',
    'ouvrir',
    'fermer',
    'commencer',
    'finir',
    # Mots de liaison et adverbes
    'puis',
    'après',
    'avant',
    'pendant',
    'soudain',
    'tout',
    'très',
    'bien',
    'mal',
    'plus',
    'moins',
    'encore',
    'déjà',
    'toujours',
    'jamais',
    'parfois',
    'souvent',
    'beaucoup',
    'peu',
    'assez',
    'trop',
    'vraiment',
    'plutôt',
    # Mots vagues
    'personne',
    'quelqu',
    'quelque',
    'quelquun',
    'part',
    'endroit',
    'lieu',
    'temps',
    'année',
    'jour',
    'heure',
    'minute',
    'seconde',
    'vie',
    'mort',
    'histoire',
    'situation',
    'problème',
    'question',
    'réponse',
    'idée',
}

# Verbes d'action spécifiques qu'on veut garder (actions significatives dans les rêves)
SIGNIFICANT_DREAM_VERBS = {
    'voler',
    'tomber',
    'courir',
    'fuir',
    'poursuivre',
    'chaser',
    'nager',
    'escalader',
    'grimper',
    'danser',
    'chanter',
    'crier',
    'pleurer',
    'rire',
    'embrasser',
    'frapper',
    'tuer',
    'mourir',
    'naître',
    'marier',
    'divorcer',
    'conduire',
    'voyager',
    'partir',
    'explorer',
    'chercher',
    'cacher',
}


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
    http_client=httpx.Client(http2=False, timeout=30),  # HTTP/1.1 + timeout ↑
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

    expected_keys = [
        "Émotionnelle",
        "Symbolique",
        "Cognitivo-scientifique",
        "Freudien",
    ]
    fixed_interpretation = {}

    logger.debug(
        f"Validation interprétation - Clés reçues: {list(interpretation_data.keys())}"
    )

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
            # Sinon, convertir en string
            else:
                fixed_interpretation[key] = str(value)
                logger.warning(
                    f"Conversion forcée en string pour {key}: {type(value)}"
                )
        else:
            # Clé manquante, ajouter un placeholder
            fixed_interpretation[key] = "Interprétation non disponible"
            logger.warning(f"Clé manquante: {key}")

    logger.debug("Validation interprétation terminée avec succès")
    return fixed_interpretation


# ---------- RETRY SYSTEM ----------


def _is_retryable_transcription_error(err: Exception) -> bool:
    """nouveau: détecte les erreurs réseau/temporaires qui méritent un retry"""
    msg = str(err).lower()
    keywords = [
        "connection error",
        "connection reset",
        "connection aborted",
        "timeout",
        "temporarily unavailable",
        "service unavailable",
        "tls",
        "ssl",
        "proxy",
        "rate limit",
        "503",
        "502",
        "429",
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
            timeout = httpx.Timeout(
                connect=15.0, read=180.0, write=60.0, pool=60.0
            )
            limits = httpx.Limits(
                max_keepalive_connections=5, max_connections=10
            )
            with httpx.Client(
                http2=False, timeout=timeout, limits=limits, trust_env=True
            ) as client:
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


# ---------- TRANSCRIPTION ----------


def transcribe_audio(audio_data, language="fr"):
    """Transcrit un audio en texte avec Whisper de Groq + système retry"""
    logger.info(f"Transcription audio démarrée - {len(audio_data)} bytes")
    start_time = time.time()

    # garde-fou si la clé est absente/mal configurée en préprod
    if not GROQ_API_KEY:
        logger.error("Échec transcription audio: GROQ_API_KEY manquante")
        return None

    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix='.wav', delete=False
        ) as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name

        last_error = None

        # Système de retry avec backoff exponentiel
        for attempt in range(1, TRANSCRIBE_MAX_RETRIES + 1):
            try:
                logger.info(
                    f"Transcription tentative {attempt}/{TRANSCRIBE_MAX_RETRIES}"
                )
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

                duration = time.time() - start_time

                # Alertes sur contenu problématique
                if len(transcription.text) < 10:
                    logger.warning(
                        f"Transcription très courte: {len(transcription.text)} caractères"
                    )

                if duration > 5:
                    logger.warning(f"Transcription lente: {duration:.2f}s")

                logger.info(
                    f"Transcription réussie - {len(transcription.text)} caractères en {duration:.2f}s"
                )
                return transcription.text

            except Exception as e:
                last_error = e
                if (
                    _is_retryable_transcription_error(e)
                    and attempt < TRANSCRIBE_MAX_RETRIES
                ):
                    sleep_s = round(TRANSCRIBE_BACKOFF_BASE**attempt, 2)
                    logger.warning(
                        f"Transcription erreur réseau (retry dans {sleep_s}s): {e}"
                    )
                    time.sleep(sleep_s)
                    continue
                else:
                    logger.error(
                        "Transcription error (%s): %s", type(e).__name__, e
                    )
                    break

        # Fallback HTTPX en dernier recours
        logger.info("Tentative fallback HTTPX pour la transcription…")
        result = _transcribe_via_httpx(temp_file_path, language)

        if result:
            duration = time.time() - start_time
            logger.info(f"Fallback HTTPX réussi en {duration:.2f}s")

        return result

    finally:
        # Nettoyage du fichier temporaire même en cas d'erreur
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(
                    f"Impossible de supprimer le fichier temporaire: {e}"
                )


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
    logger.info(f"[{operation}] Démarrage avec {model}")
    start_time = time.time()

    # Hiérarchie de fallback par modèle
    fallback_chain = {
        "mistral-large-latest": [
            "mistral-medium",
            "mistral-small-latest",
            "open-mistral-7b",
        ],
        "mistral-medium": ["mistral-small-latest", "open-mistral-7b"],
        "mistral-small-latest": ["open-mistral-7b"],
        "open-mistral-7b": [],
    }

    models_to_try = [model] + fallback_chain.get(model, [])
    logger.debug(f"[{operation}] Chaîne de fallback: {models_to_try}")

    for attempt, current_model in enumerate(models_to_try):
        try:
            attempt_start = time.time()
            response = mistral_client.chat.complete(
                model=current_model,
                messages=messages,
                response_format={"type": "json_object"},
            )
            attempt_duration = time.time() - attempt_start

            if attempt > 0:
                logger.warning(
                    f"[{operation}] Fallback utilisé: {current_model} en {attempt_duration:.2f}s"
                )
            else:
                logger.info(
                    f"[{operation}] Succès avec {current_model} en {attempt_duration:.2f}s"
                )

            # Alerte sur performance dégradée
            if attempt_duration > 10:
                logger.warning(
                    f"[{operation}] Performance dégradée: {attempt_duration:.2f}s"
                )

            return response

        except Exception as e:
            error_msg = str(e).lower()
            attempt_duration = time.time() - attempt_start

            # Erreurs qui nécessitent un fallback
            if any(
                keyword in error_msg
                for keyword in [
                    "insufficient_quota",
                    "quota_exceeded",
                    "rate_limit",
                    "model_not_found",
                    "service_unavailable",
                    "timeout",
                ]
            ):
                if "quota" in error_msg:
                    logger.warning(
                        f"[{operation}] QUOTA ATTEINT - {current_model}"
                    )
                elif "rate_limit" in error_msg:
                    logger.warning(
                        f"[{operation}] RATE LIMIT - {current_model}"
                    )
                else:
                    logger.warning(
                        f"[{operation}] Erreur {current_model}: {e}"
                    )

                if attempt == len(models_to_try) - 1:
                    total_duration = time.time() - start_time
                    logger.error(
                        f"[{operation}] Tous les fallbacks échoués après {total_duration:.2f}s"
                    )
                    return None

                continue
            else:
                logger.error(
                    f"[{operation}] Erreur critique {current_model}: {e}"
                )
                raise e

    return None


# ---------- ANALYSE D'ÉMOTIONS ----------


def analyze_emotions(text):
    """Renvoie le score des émotions + l'émotion dominante avec fallback"""
    logger.info(f"Analyse émotionnelle démarrée - {len(text)} caractères")

    system_prompt = read_file("context_emotion.txt")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]

    response = safe_mistral_call(
        model="mistral-small-latest",
        messages=messages,
        operation="Analyse émotionnelle",
    )

    if response is None:
        logger.error(
            "Échec analyse émotionnelle - tous les modèles indisponibles"
        )
        return None, None

    try:
        # Contrôle de format robuste
        raw = json.loads(response.choices[0].message.content)

        # Certains modèles peuvent renvoyer une liste de paires; on la convertit en dict si possible
        if isinstance(raw, list):
            try:
                raw = dict(raw)
            except Exception:
                logger.error(
                    f"Format inattendu des émotions (liste non convertible): {raw}"
                )
                return None, None

        if not isinstance(raw, dict):
            logger.error(f"Format inattendu des émotions (type={type(raw)})")
            return None, None

        # Cast des valeurs non numériques
        cleaned = {}
        for k, v in raw.items():
            try:
                cleaned[k] = float(v)
            except (TypeError, ValueError):
                logger.warning(f"Score non numérique ignoré pour {k}: {v}")

        if not cleaned:
            logger.error("Aucun score exploitable reçu")
            return None, None

        scores = softmax(cleaned)
        dominant = max(scores.items(), key=lambda x: x[1])

        logger.info(f"Émotion dominante: {dominant[0]} ({dominant[1]:.2f})")
        logger.debug(f"Scores détaillés: {json.dumps(scores, indent=2)}")

        return scores, dominant

    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Erreur parsing émotions: {e}")
        return None, None


def classify_dream(emotions):
    """Détermine si le rêve est un cauchemar ou non"""
    if emotions is None:
        logger.warning("Classification impossible - émotions non disponibles")
        return None

    with open(
        os.path.join(BASE_DIR, "diary", "prompt", "reference_emotions.json")
    ) as f:
        ref = json.load(f)

    pos = [emotions[e] for e in ref["positif"] if e in emotions]
    neg = [emotions[e] for e in ref["negatif"] if e in emotions]

    avg_pos = sum(pos) / len(pos or [1])
    avg_neg = sum(neg) / len(neg or [1])

    classification = "cauchemar" if avg_neg > avg_pos else "rêve"
    logger.info(f"Classification: {classification}")
    logger.debug(f"Scores - positif: {avg_pos:.2f}, négatif: {avg_neg:.2f}")

    return classification


# ---------- INTERPRÉTATION ----------


def interpret_dream(text):
    """Demande à Mistral une interprétation du rêve avec fallback et validation"""
    logger.info(f"Interprétation démarrée - {len(text)} caractères")

    system_prompt = read_file("context_interpretation.txt")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]

    response = safe_mistral_call(
        model="mistral-large-latest",
        messages=messages,
        operation="Interprétation",
    )

    if response is None:
        logger.error("Échec interprétation - tous les modèles indisponibles")
        return None

    try:
        raw_interpretation = json.loads(response.choices[0].message.content)
        logger.debug("Réponse IA reçue, validation en cours...")

        # Valider et corriger le format
        validated_interpretation = validate_and_fix_interpretation(
            raw_interpretation
        )

        if validated_interpretation:
            logger.info("Interprétation générée avec succès")
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
    logger.info(f"Génération image pour rêve {dream_instance.id}")
    start_time = time.time()

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
                agent_id=agent.id, inputs=prompt_text
            )

            file_id = next(
                (
                    item.file_id
                    for output in conversation.outputs
                    if hasattr(output, "content")
                    for item in output.content
                    if hasattr(item, "file_id")
                ),
                None,
            )

            if not file_id:
                logger.warning("Aucune image générée par l'agent")
                return False

            image_bytes = mistral_client.files.download(file_id=file_id).read()

            # Stocker en base64 au lieu de fichier
            dream_instance.set_image_from_bytes(image_bytes, format='PNG')
            dream_instance.save()

            duration = time.time() - start_time
            logger.info(f"Image générée avec succès en {duration:.2f}s")
            return True

        except Exception as e:
            error_msg = str(e).lower()
            if any(
                keyword in error_msg
                for keyword in [
                    "insufficient_quota",
                    "quota_exceeded",
                    "rate_limit",
                ]
            ):
                logger.warning(f"Quota image atteint: {e}")
                return False
            else:
                raise e

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Erreur génération image après {duration:.2f}s: {e}")
        return False


# ---------- THEMATIQUE ----------


def _extract_significant_words(text: str) -> List[str]:
    """Extrait uniquement les mots potentiellement significatifs"""
    if not nlp or not text:
        return _fallback_extract_words(text)

    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)

    doc = nlp(text)
    words = []

    for token in doc:
        lemma = token.lemma_.lower()

        # Garder principalement les noms et quelques verbes/adjectifs spécifiques
        if (
            token.pos_ in ['NOUN', 'PROPN']
            or (token.pos_ == 'VERB' and lemma in SIGNIFICANT_DREAM_VERBS)
            or (token.pos_ == 'ADJ' and len(lemma) >= 5)
        ):

            if (
                len(lemma) >= 3
                and lemma not in FRENCH_STOPWORDS
                and lemma not in DREAM_SPECIFIC_STOPWORDS
                and not lemma.isdigit()
                and token.is_alpha
            ):

                words.append(lemma)

    return words


def _fallback_extract_words(text: str) -> List[str]:
    """Version fallback sans spaCy"""
    if not text:
        return []

    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = text.split()

    words = []
    for token in tokens:
        if (
            len(token) >= 4
            and token not in FRENCH_STOPWORDS
            and token not in DREAM_SPECIFIC_STOPWORDS
            and not token.isdigit()
            and token.isalpha()
        ):

            if stemmer:
                stemmed = stemmer.stem(token)
                if len(stemmed) >= 3:
                    words.append(stemmed)
            else:
                words.append(token)

    return words


def _categorize_words_by_theme(words: List[str]) -> dict:
    """Catégorise les mots par thèmes"""
    theme_matches = {}

    for theme_name, keywords in THEME_CATEGORIES.items():
        matching_words = []

        for word in words:
            for keyword in keywords:
                # Correspondance flexible : exacte ou inclusion partielle
                if (
                    word == keyword
                    or keyword in word
                    or word in keyword
                    or (
                        stemmer and stemmer.stem(word) == stemmer.stem(keyword)
                    )
                ):
                    matching_words.append(word)
                    break

        if matching_words:
            theme_matches[theme_name] = len(set(matching_words))

    return theme_matches


def _calculate_theme_document_frequency(dream_texts: List[str]) -> Counter:
    """Calcule la fréquence des thèmes par document (rêve)"""
    theme_document_freq = Counter()

    for dream_text in dream_texts:
        words = _extract_significant_words(dream_text)
        theme_matches = _categorize_words_by_theme(words)

        # Marquer la présence de chaque thème dans ce rêve
        for theme_name, match_count in theme_matches.items():
            if match_count > 0:  # Au moins un mot de cette catégorie
                theme_document_freq[theme_name] += 1

    return theme_document_freq


def analyze_recurring_themes(user, min_dreams=2, min_occurrence=2):
    """Analyse les thématiques récurrentes par catégories conceptuelles"""
    logger.info(f"Analyse thématiques récurrentes user {user.id}")

    dreams = (
        Dream.objects.filter(user=user, transcription__isnull=False)
        .exclude(transcription="")
        .values_list('transcription', flat=True)
    )

    dream_texts = list(dreams)
    total_dreams = len(dream_texts)

    if total_dreams < min_dreams:
        return {
            'top_theme': 'Pas encore de données',
            'percentage': 0,
            'total_dreams': total_dreams,
            'message': f'Au moins {min_dreams} rêves nécessaires',
        }

    # Calculer la fréquence documentaire des thèmes
    theme_freq = _calculate_theme_document_frequency(dream_texts)

    # Filtrer les thèmes récurrents
    recurring_themes = [
        (theme, freq)
        for theme, freq in theme_freq.items()
        if freq >= min_occurrence
    ]

    if not recurring_themes:
        return {
            'top_theme': 'Aucune récurrence détectée',
            'percentage': 0,
            'total_dreams': total_dreams,
            'message': 'Pas de thématique récurrente trouvée',
        }

    # Trier par fréquence décroissante
    recurring_themes.sort(key=lambda x: x[1], reverse=True)

    # Prendre le thème principal
    top_theme_name, top_theme_count = recurring_themes[0]
    top_theme_percentage = round((top_theme_count / total_dreams) * 100, 1)

    logger.info(
        f"Thème récurrent détecté: {top_theme_name} ({top_theme_percentage}%)"
    )

    return {
        'top_theme': top_theme_name.capitalize(),
        'percentage': top_theme_percentage,
        'total_dreams': total_dreams,
        'all_themes': recurring_themes[:10],
        'message': f'{len(recurring_themes)} thématiques récurrentes trouvées',
    }


# ---------- PROFIL ONYRIQUE ----------


def get_profil_onirique_stats(user):
    """Calcule les statistiques du profil onirique d'un utilisateur"""
    logger.info(f"Calcul profil onirique user {user.id}")

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
            "thematique_recurrente": "Pas encore de données",
            "thematique_percentage": 0,
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
        logger.info(
            f"Profil calculé: {statut_reveuse} ({pourcentage}%), émotion: {emotion_dominante}"
        )
    else:
        emotion_dominante = "émotion endormie"
        emotion_percentage = 0

    # Analyse thématique
    theme_analysis = analyze_recurring_themes(user)

    return {
        "statut_reveuse": statut_reveuse,
        "pourcentage_reveuse": pourcentage,
        "label_reveuse": label,
        "emotion_dominante": emotion_dominante,
        "emotion_dominante_percentage": emotion_percentage,
        "thematique_recurrente": theme_analysis['top_theme'],
        "thematique_percentage": theme_analysis['percentage'],
    }


# ---------- DASHBOARD PERSONNEL ----------


def get_date_filter_queryset(
    user, period=None, start_date=None, end_date=None
):
    """
    Retourne un queryset filtré selon la période choisie

    Args:
        user: L'utilisateur
        period: 'month', '3months', '6months', '1year', 'all' ou None
        start_date: Date de début personnalisée (format YYYY-MM-DD)
        end_date: Date de fin personnalisée (format YYYY-MM-DD)
    """
    queryset = Dream.objects.filter(user=user)

    # Si dates personnalisées fournies
    if start_date and end_date:
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d').date()
            end = datetime.strptime(end_date, '%Y-%m-%d').date()
            queryset = queryset.filter(created_at__date__range=[start, end])
            logger.debug(f"Filtre personnalisé: {start} à {end}")
            return queryset
        except ValueError:
            logger.warning(f"Dates invalides: {start_date}, {end_date}")
            # En cas d'erreur, on continue avec le period

    # Filtres prédéfinis
    if period == 'month':
        start_date = timezone.now() - timedelta(days=30)
        queryset = queryset.filter(created_at__gte=start_date)
        logger.debug("Filtre: 30 derniers jours")
    elif period == '3months':
        start_date = timezone.now() - timedelta(days=90)
        queryset = queryset.filter(created_at__gte=start_date)
        logger.debug("Filtre: 3 derniers mois")
    elif period == '6months':
        start_date = timezone.now() - timedelta(days=180)
        queryset = queryset.filter(created_at__gte=start_date)
        logger.debug("Filtre: 6 derniers mois")
    elif period == '1year':
        start_date = timezone.now() - timedelta(days=365)
        queryset = queryset.filter(created_at__gte=start_date)
        logger.debug("Filtre: 1 an")
    else:
        logger.debug("Aucun filtre appliqué")

    return queryset


def get_dream_type_stats_filtered(
    user, period=None, start_date=None, end_date=None
):
    """Version filtrée de get_dream_type_stats"""
    dreams = get_date_filter_queryset(user, period, start_date, end_date)
    total = dreams.count()

    if total == 0:
        return {
            'percentages': {'rêve': 0, 'cauchemar': 0},
            'counts': {'rêve': 0, 'cauchemar': 0},
            'total': 0,
        }

    nb_reves = dreams.filter(dream_type='rêve').count()
    nb_cauchemars = dreams.filter(dream_type='cauchemar').count()

    return {
        'percentages': {
            'rêve': round((nb_reves / total) * 100, 1),
            'cauchemar': round((nb_cauchemars / total) * 100, 1),
        },
        'counts': {'rêve': nb_reves, 'cauchemar': nb_cauchemars},
        'total': total,
    }


def get_dream_type_timeline_filtered(
    user, period=None, start_date=None, end_date=None
):
    """Version filtrée de get_dream_type_timeline"""
    dreams = (
        get_date_filter_queryset(user, period, start_date, end_date)
        .annotate(date_only=TruncDate('created_at'))
        .values('date_only', 'dream_type')
        .annotate(count=Count('id'))
        .order_by('date_only')
    )

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
        timeline_list.append(
            {
                'date': date_str,
                'rêve': counts['rêve'],
                'cauchemar': counts['cauchemar'],
            }
        )

    return timeline_list


def get_emotions_stats_filtered(
    user, period=None, start_date=None, end_date=None
):
    """Version filtrée de get_emotions_stats"""
    dreams = get_date_filter_queryset(
        user, period, start_date, end_date
    ).exclude(dominant_emotion__isnull=True)
    total = dreams.count()

    if total == 0:
        return {'percentages': {}, 'counts': {}, 'total': 0}

    emotion_counts = Counter(dreams.values_list('dominant_emotion', flat=True))

    # Calculer les pourcentages
    emotion_percentages = {}
    for emotion, count in emotion_counts.items():
        emotion_percentages[emotion] = round((count / total) * 100, 1)

    return {
        'percentages': emotion_percentages,
        'counts': dict(emotion_counts),
        'total': total,
    }


def get_emotions_timeline_filtered(
    user, period=None, start_date=None, end_date=None
):
    """Version filtrée de get_emotions_timeline"""
    dreams = (
        get_date_filter_queryset(user, period, start_date, end_date)
        .exclude(dominant_emotion__isnull=True)
        .annotate(date_only=TruncDate('created_at'))
        .values('date_only', 'dominant_emotion')
        .annotate(count=Count('id'))
        .order_by('date_only')
    )

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
