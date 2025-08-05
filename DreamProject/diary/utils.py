import os
import json
import math
import tempfile
from datetime import datetime
from django.core.files.base import ContentFile
from dotenv import load_dotenv
from django.conf import settings
from groq import Groq
from mistralai import Mistral
from collections import Counter
from .models import Dream

# Chargement des variables d’environnement
load_dotenv()

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

# ---------- TRANSCRIPTION ----------

def transcribe_audio(audio_data, language="fr"):
    """Transcrit un audio en texte avec Whisper de Groq"""
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name

        with open(temp_file_path, "rb") as audio_file:
            transcription = groq_client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3-turbo",
                prompt="Specify context or spelling",
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"],
                language=language,
                temperature=0.0,
            )

        os.unlink(temp_file_path)
        return transcription.text

    except Exception:
        return None

# ---------- ANALYSE D'ÉMOTIONS ----------

def analyze_emotions(text):
    """Renvoie le score des émotions + l’émotion dominante"""
    system_prompt = read_file("context_emotion.txt")
    response = mistral_client.chat.complete(
        model="mistral-medium",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
    )
    scores = softmax(json.loads(response.choices[0].message.content))
    dominant = max(scores.items(), key=lambda x: x[1])
    return scores, dominant

def classify_dream(emotions):
    """Détermine si le rêve est un cauchemar ou non"""
    with open(os.path.join(BASE_DIR, "diary", "prompt", "reference_emotions.json")) as f:
        ref = json.load(f)
    pos = [emotions[e] for e in ref["positif"] if e in emotions]
    neg = [emotions[e] for e in ref["negatif"] if e in emotions]
    return "cauchemar" if sum(neg) / len(neg or [1]) > sum(pos) / len(pos or [1]) else "rêve"

# ---------- INTERPRÉTATION ----------

def interpret_dream(text):
    """Demande à Mistral une interprétation du rêve"""
    system_prompt = read_file("context_interpretation.txt")
    resp = mistral_client.chat.complete(
        model="mistral-medium",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)

# ---------- IMAGE ----------

def generate_image_from_text(user, prompt_text, dream_instance):
    """
    Génère une image IA à partir du texte du rêve, via agent Mistral.
    Attache l’image et le prompt d’entrée au modèle Dream.
    """
    try:
        system_instructions = read_file("instructions_image.txt")

        agent = mistral_client.beta.agents.create(
            model="mistral-medium-2505",
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
            return False

        image_bytes = mistral_client.files.download(file_id=file_id).read()
        filename = f"dream_{dream_instance.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        dream_instance.image.save(filename, ContentFile(image_bytes))
        dream_instance.image_prompt = prompt_text
        dream_instance.save()

        return True

    except Exception:
        return False
    


def get_profil_onirique_stats(user):
    dreams = Dream.objects.filter(user=user)
    total = dreams.count()

    # Statut rêveuse ou non
    nb_reves = dreams.filter(dream_type='rêve').count()
    nb_cauchemars = dreams.filter(dream_type='cauchemar').count()

    if nb_reves >= nb_cauchemars:
        statut_reveuse = "rêveuse"
        pourcentage = round((nb_reves / total) * 100) if total else 0
        label = "rêves"
    else:
        statut_reveuse = "en proie aux cauchemars"
        pourcentage = round((nb_cauchemars / total) * 100) if total else 0
        label = "cauchemars"

    # Émotion dominante
    emotions = dreams.values_list('dominant_emotion', flat=True)
    emotion_counts = Counter(emotions)
    if emotion_counts:
        emotion_dominante, count = emotion_counts.most_common(1)[0]
        emotion_percentage = round((count / total) * 100) if total else 0
    else:
        emotion_dominante = "Non définie"
        emotion_percentage = 0

    return {
        "statut_reveuse": statut_reveuse,
        "pourcentage_reveuse": pourcentage,
        "label_reveuse": label,
        "emotion_dominante": emotion_dominante,
        "emotion_dominante_percentage": emotion_percentage,
    }

