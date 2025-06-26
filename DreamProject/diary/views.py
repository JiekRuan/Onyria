import os
import json
from dotenv import load_dotenv
from django.shortcuts import render, get_object_or_404, reverse
from mistralai import Mistral
import math
from datetime import datetime
from .models import Dream


from django.contrib.auth.decorators import login_required
# Create your views here.

load_dotenv()

api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    raise ValueError("MISTRAL_API_KEY est manquant dans .env")

mistral_client = Mistral(api_key=api_key)

def read_file(filename):
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, filename)
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def classify_dream_from_emotions(emotions):
    base_dir = os.path.dirname(__file__)
    ref_path = os.path.join(base_dir, "reference_emotions.json")

    with open(ref_path, "r", encoding="utf-8") as file:
        reference_emotions_dict = json.load(file)

    positif = reference_emotions_dict["positif"]
    negatif = reference_emotions_dict["negatif"]

    pos_vals = [emotions[e] for e in positif if e in emotions]
    neg_vals = [emotions[e] for e in negatif if e in emotions]

    score_positif = sum(pos_vals) / len(pos_vals) if pos_vals else 0
    score_negatif = sum(neg_vals) / len(neg_vals) if neg_vals else 0

    if score_negatif > score_positif:
        return "cauchemar"
    else:
        return "rêve"

def analyser_texte_view(request):
    result = None
    dominant_emotion = None
    dream_type = None
    image_path = None
    interpretation = None
    user_text = ""

    if request.method == "POST":
        user_text = request.POST.get("texte", "")
        if user_text:
            # Analyse des émotions
            chat_response = mistral_client.chat.complete(
                model="mistral-large-latest",
                messages=[
                    {"role": "system", "content": read_file("context_emotion.txt")},
                    {"role": "user", "content": user_text},
                ],
                response_format={"type": "json_object"},
            )

            result = json.loads(chat_response.choices[0].message.content)

            if result:
                dominant_emotion = max(result.items(), key=lambda x: x[1])
                dream_type = classify_dream_from_emotions(result)

            # Interprétation du rêve
            interpretation = interpret_dream_with_ai(user_text)

            # Image (optionnel)
            # image_path = prompt_to_image(user_text)

    return render(request, "diary/models.html", {
        "result": result,
        "dominant_emotion": dominant_emotion,
        "dream_type": dream_type,
        "interpretation": interpretation,
        "image_path": image_path,
        "texte": user_text,
    })


def interpret_dream_with_ai(dream_text):
    chat_response = mistral_client.chat.complete(
        model="mistral-large-latest",
        messages=[
            {
                "role": "system",
                "content": read_file("context_interpretation.txt"),
            },
            {
                "role": "user",
                "content": dream_text,
            },
        ],
        response_format={"type": "json_object"},
    )

    return json.loads(chat_response.choices[0].message.content)

def prompt_to_image(prompt):
    mistral_image_agent = mistral_client.beta.agents.create(
        model="mistral-medium-2505",
        name="Image Generation Agent",
        description="Agent used to generate images.",
        instructions="Use the image generation tool when you have to create images.",
        tools=[{"type": "image_generation"}],
        completion_args={
            "temperature": 0.3,
            "top_p": 0.95,
        },
    )

    response = mistral_client.beta.conversations.start(
        agent_id=mistral_image_agent.id, inputs=prompt
    )

    file_id = response.outputs[1].content[1].file_id
    file_bytes = mistral_client.files.download(file_id=file_id).read()

    output_dir = os.path.join("diary", "static", "diary", "generated_images")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"dream_image_{timestamp}.png"
    file_path = os.path.join(output_dir, filename)

    with open(file_path, "wb") as f:
        f.write(file_bytes)

    return f"diary/generated_images/{filename}" 

@login_required
def index(request):
    dreams_history = Dream.objects.filter(user=request.user)
    context = {"List of dreams": dreams_history}
    return render(request, "diary/index.html", context)

