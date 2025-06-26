import os
import json
import math
from datetime import datetime
from dotenv import load_dotenv

from django.shortcuts import render

from groq import Groq
from mistralai import Mistral

load_dotenv()

groq_client = Groq()
mistral_client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])


def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def softmax(predictions):
    exp_values = {k: math.exp(v) for k, v in predictions.items()}
    total = sum(exp_values.values())
    return {k: v / total for k, v in exp_values.items()}

def create_transcription(file, language="fr"):
    return groq_client.audio.transcriptions.create(
        file=file,
        model="whisper-large-v3-turbo",
        prompt="Specify context or spelling",
        response_format="verbose_json",
        timestamp_granularities=["word", "segment"],
        language=language,
        temperature=0.0,
    )

def speech_to_Text(file, file_type="file", language="fr", path=None):
    if file_type == "file":
        transcription = create_transcription(file)
    else:
        with open(file, "rb") as file:
            transcription = create_transcription(file)
    return transcription.text

def text_analysis(text):
    chat_response = mistral_client.chat.complete(
        model="mistral-large-latest",
        messages=[
            {"role": "system", "content": read_file("context_emotion.txt")},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
    )
    emotions = json.loads(chat_response.choices[0].message.content)
    return softmax(emotions)

def text_to_prompt(dream_text):
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {"role": "system", "content": read_file("resume_text.txt")},
            {"role": "user", "content": dream_text},
        ],
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content

def prompt_to_image(prompt):
    mistral_image_agent = mistral_client.beta.agents.create(
        model="mistral-medium-2505",
        name="Image Generation Agent",
        description="Agent used to generate images.",
        instructions="Use the image generation tool when you have to create images.",
        tools=[{"type": "image_generation"}],
        completion_args={"temperature": 0.3, "top_p": 0.95},
    )
    response = mistral_client.beta.conversations.start(
        agent_id=mistral_image_agent.id, inputs=prompt
    )

    file_id = response.outputs[1].content[1].file_id
    file_bytes = mistral_client.files.download(file_id=file_id).read()

    output_dir = "./generated_images"
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"dream_image_{timestamp}.png"
    file_path = os.path.join(output_dir, filename)

    with open(file_path, "wb") as f:
        f.write(file_bytes)

    return f"/generated_images/{filename}"  

def interpret_dream_with_ai(dream_text):
    chat_response = mistral_client.chat.complete(
        model="mistral-large-latest",
        messages=[
            {"role": "system", "content": read_file("context_interpretation.txt")},
            {"role": "user", "content": dream_text},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(chat_response.choices[0].message.content)

def classify_dream_from_emotions(emotions):
    with open('reference_emotions.json', encoding="utf-8") as file:
        reference_emotions_dict = json.load(file)
    positif = reference_emotions_dict["positif"]
    negatif = reference_emotions_dict["negatif"]

    pos_vals = [emotions[e] for e in positif if e in emotions]
    neg_vals = [emotions[e] for e in negatif if e in emotions]

    score_positif = sum(pos_vals) / len(pos_vals) if pos_vals else 0
    score_negatif = sum(neg_vals) / len(neg_vals) if neg_vals else 0

    return "cauchemar" if score_negatif > score_positif else "rêve"


def home(request):
    return render(request, 'diary/index.html')

def transcription_view(request):
    if request.method == "POST":
        audio_file = request.FILES.get('file')
        if not audio_file:
            return render(request, "error.html", {"message": "Fichier audio manquant"})
        try:
            transcription = speech_to_Text(audio_file, file_type="file")
            return render(request, "result_transcription.html", {"transcription": transcription})
        except Exception as e:
            return render(request, "error.html", {"message": str(e)})
    return render(request, "upload_audio.html")

def analyse_emotions_view(request):
    if request.method == "POST":
        text = request.POST.get("text")
        if not text:
            return render(request, "error.html", {"message": "Texte manquant"})
        try:
            emotions = text_analysis(text)
            classification = classify_dream_from_emotions(emotions)
            return render(request, "result_emotions.html", {
                "emotions": emotions,
                "classification": classification,
            })
        except Exception as e:
            return render(request, "error.html", {"message": str(e)})
    return render(request, "form_emotions.html")

def generate_prompt_view(request):
    if request.method == "POST":
        dream_text = request.POST.get("text")
        if not dream_text:
            return render(request, "error.html", {"message": "Texte manquant"})
        try:
            prompt = text_to_prompt(dream_text)
            return render(request, "result_prompt.html", {"prompt": prompt})
        except Exception as e:
            return render(request, "error.html", {"message": str(e)})
    return render(request, "form_prompt.html")

def generate_image_view(request):
    if request.method == "POST":
        prompt = request.POST.get("prompt")
        if not prompt:
            return render(request, "error.html", {"message": "Prompt manquant"})
        try:
            image_path = prompt_to_image(prompt)
            return render(request, "result_image.html", {"image_path": image_path})
        except Exception as e:
            return render(request, "error.html", {"message": str(e)})
    return render(request, "form_image.html")

def interpret_dream_view(request):
    if request.method == "POST":
        dream_text = request.POST.get("text")
        if not dream_text:
            return render(request, "error.html", {"message": "Texte manquant"})
        try:
            interpretation = interpret_dream_with_ai(dream_text)
            return render(request, "result_interpretation.html", {"interpretation": interpretation})
        except Exception as e:
            return render(request, "error.html", {"message": str(e)})
    return render(request, "form_interpretation.html")
