# engine/main.py
import os
import torch
import opensmile
from dotenv import load_dotenv
from groq import Groq
from transformers import AutoTokenizer, AutoModel

from loop.redis_loop import start_redis_loop
from inference.run_model import load_snn_model

load_dotenv()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Groq (UNCHANGED)
# ----------------------------
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def analyze_sentiment(text):
    """
    Analyze the emotional sentiment of the text using Groq API.
    Returns a short response message.
    """
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an emotional sentiment analysis chatbot. "
                    "Analyze the user's message and respond with a short, "
                    "empathetic message (1-2 sentences) that acknowledges "
                    "their emotional state and provides supportive feedback."
                )
            },
            {
                "role": "user",
                "content": text
            }
        ],
        model="llama-3.1-8b-instant",
        temperature=0.7,
        max_tokens=100
    )
    return chat_completion.choices[0].message.content


# ----------------------------
# Load pretrained models ONCE
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(
    "j-hartmann/emotion-english-distilroberta-base"
)
text_model = AutoModel.from_pretrained(
    "j-hartmann/emotion-english-distilroberta-base"
).to(DEVICE).eval()

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)

snn_model = load_snn_model(
    "snn/spikemo_best_IEMOCAP.pt",
    device=DEVICE
)

# ----------------------------
# Start Redis loop
# ----------------------------
if __name__ == "__main__":
    start_redis_loop(
        snn_model=snn_model,
        tokenizer=tokenizer,
        text_model=text_model,
        smile=smile,
        groq_fn=analyze_sentiment,
        device=DEVICE
    )