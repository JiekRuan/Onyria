import os, json, math
from django.shortcuts import render
from django.conf import settings
from dotenv import load_dotenv
from mistralai import Mistral
from datetime import datetime

load_dotenv()
BASE_DIR = settings.BASE_DIR

mistral_client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

def read_file(file_path):
    full_path = os.path.join(BASE_DIR, "diary", "prompt", file_path)
    with open(full_path, "r", encoding="utf-8") as f:
        return f.read()

def softmax(preds):
    exp = {k: math.exp(v) for k, v in preds.items()}
    total = sum(exp.values())
    return {k: v / total for k, v in exp.items()}

def classify_dream_from_emotions(emotions):
    with open(os.path.join(BASE_DIR, "diary", "prompt", "reference_emotions.json")) as f:
        ref = json.load(f)
    pos_vals = [emotions[e] for e in ref["positif"] if e in emotions]
    neg_vals = [emotions[e] for e in ref["negatif"] if e in emotions]
    score_pos = sum(pos_vals) / len(pos_vals) if pos_vals else 0
    score_neg = sum(neg_vals) / len(neg_vals) if neg_vals else 0
    return "cauchemar" if score_neg > score_pos else "rêve"

def interpret_dream_with_ai(dream_text):
    resp = mistral_client.chat.complete(
        model="mistral-large-latest",
        messages=[
            {"role": "system", "content": read_file("context_interpretation.txt")},
            {"role": "user", "content": dream_text},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)

def prompt_to_image(prompt):
    try:
        # Création de l'agent image
        mistral_image_agent = mistral_client.beta.agents.create(
            model="mistral-medium-2505",
            name="Dream Image Agent",
            instructions=read_file("instructions_image.txt"),
            tools=[{"type": "image_generation"}],
            completion_args={
                "temperature": 0.3,
                "top_p": 0.95,
            },
        )

        # Lancer la génération
        conversation = mistral_client.beta.conversations.start(
            agent_id=mistral_image_agent.id, inputs=prompt
        )

        file_id = None
        for output in conversation.outputs:
            if hasattr(output, "content"):
                for item in output.content:
                    if hasattr(item, "file_id"):
                        file_id = item.file_id
                        break

        if not file_id:
            return None

        file_bytes = mistral_client.files.download(file_id=file_id).read()

        # Sauvegarde
        output_dir = os.path.join(BASE_DIR, "diary", "static", "diary", "generated_images")
        os.makedirs(output_dir, exist_ok=True)

        filename = f"dream_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        file_path = os.path.join(output_dir, filename)

        with open(file_path, "wb") as f:
            f.write(file_bytes)

        return f"diary/generated_images/{filename}"

    except Exception:
        return None


def analyser_texte_view(request):
    result = None
    dominant_emotion = None
    dream_type = None
    interpretation = None
    image_path = None
    user_text = ""

    if request.method == "POST":
        user_text = request.POST.get("texte", "")

        # Analyse émotions
        emotions_resp = mistral_client.chat.complete(
            model="mistral-large-latest",
            messages=[
                {"role": "system", "content": read_file("context_emotion.txt")},
                {"role": "user", "content": user_text},
            ],
            response_format={"type": "json_object"},
        )
        result = softmax(json.loads(emotions_resp.choices[0].message.content))
        dominant_emotion = max(result.items(), key=lambda x: x[1])
        dream_type = classify_dream_from_emotions(result)
        interpretation = interpret_dream_with_ai(user_text)

        # Génération du prompt
        prompt_resp = mistral_client.chat.complete(
            model="mistral-large-latest",
            messages=[
                {"role": "system", "content": read_file("resume_text.txt")},
                {"role": "user", "content": user_text},
            ],
        )
        prompt = prompt_resp.choices[0].message.content
        image_path = prompt_to_image(prompt)

    return render(request, "diary/models.html", {
        "result": result,
        "dominant_emotion": dominant_emotion,
        "dream_type": dream_type,
        "interpretation": interpretation,
        "image_path": image_path,
        "texte": user_text,
    })

