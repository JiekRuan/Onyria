import os
import json
import math
import tempfile
from datetime import datetime
from dotenv import load_dotenv
from django.conf import settings
from groq import Groq
from mistralai import Mistral

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

    except Exception as e:
        print(f"Erreur: {e}")
        return None

# ---------- ANALYSE D'ÉMOTIONS ----------

def analyze_emotions(text):
    """Renvoie le score des émotions + l’émotion dominante"""
    response = mistral_client.chat.complete(
        model="mistral-large-latest",
        messages=[
            {"role": "system", "content": read_file("context_emotion.txt")},
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
    resp = mistral_client.chat.complete(
        model="mistral-large-latest",
        messages=[
            {"role": "system", "content": read_file("context_interpretation.txt")},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)

# ---------- IMAGE ----------

def generate_image_from_text(text):
    """Génère une image à partir d’un résumé de rêve"""
    try:
        prompt_resp = mistral_client.chat.complete(
            model="mistral-large-latest",
            messages=[
                {"role": "system", "content": read_file("resume_text.txt")},
                {"role": "user", "content": text},
            ],
        )
        prompt = prompt_resp.choices[0].message.content

        agent = mistral_client.beta.agents.create(
            model="mistral-medium-2505",
            name="Dream Image Agent",
            instructions=read_file("instructions_image.txt"),
            tools=[{"type": "image_generation"}],
            completion_args={"temperature": 0.3, "top_p": 0.95},
        )

        conversation = mistral_client.beta.conversations.start(
            agent_id=agent.id, inputs=prompt
        )

        file_id = next(
            (item.file_id for output in conversation.outputs if hasattr(output, "content")
             for item in output.content if hasattr(item, "file_id")),
            None
        )

        if not file_id:
            return None

        image_bytes = mistral_client.files.download(file_id=file_id).read()
        output_dir = os.path.join(BASE_DIR, "diary", "static", "diary", "generated_images")
        os.makedirs(output_dir, exist_ok=True)
        filename = f"dream_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        path = os.path.join(output_dir, filename)

        with open(path, "wb") as f:
            f.write(image_bytes)

        return f"diary/generated_images/{filename}"

    except Exception:
        return None