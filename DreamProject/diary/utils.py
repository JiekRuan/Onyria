import os
import tempfile
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# Client Groq
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def create_transcription(file, language="fr"):
    transcription = groq_client.audio.transcriptions.create(
        file=file,
        model="whisper-large-v3-turbo",
        prompt="Specify context or spelling",
        response_format="verbose_json",
        timestamp_granularities=["word", "segment"],
        language=language,
        temperature=0.0,
    )
    return transcription

def transcribe_audio(audio_data, language="fr"):
    """Transcrit un fichier audio"""
    try:
        # Fichier temporaire
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name
        
        # Transcription
        with open(temp_file_path, "rb") as audio_file:
            transcription = create_transcription(audio_file, language)
        
        # Nettoyage
        os.unlink(temp_file_path)
        return transcription.text
        
    except Exception as e:
        print(f"Erreur: {e}")
        return None