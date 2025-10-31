import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torchaudio
import numpy as np
import snntorch.spikegen as spikegen
import io
from spikingjelly.activation_based import functional
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import logout
from .api import register_routes
from .database.core import engine, Base

# Import your efficient SpikeNet model
from .model import SpikeNetEfficient as SpikeNet


# ---------------- FASTAPI SETUP ----------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base.metadata.create_all(bind=engine)


register_routes(app)



# ---------------- DATA MODELS ----------------
class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    emotion: str


# ---------------- MODEL CONFIG ----------------
MODEL_PATH = "snn_best_model.pth"
IN_DIM = 1611
NUM_CLASSES = 7
EMBED_DIM = 256
T = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LOAD SNN ----------------
print("🔹 Loading emotion detection SNN...")
model = SpikeNet(
    in_dim=IN_DIM,
    embed_dim=EMBED_DIM,
    num_classes=NUM_CLASSES,
    T=T
).to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

label_map = {
    0: "Neutral",
    1: "Happy",
    2: "Sad",
    3: "Angry",
    4: "Fearful",
    5: "Disgust",
    6: "Surprised",
}


# ---------------- LOAD LOCAL LLM ----------------
print("🔹 Loading local LLM model (Phi-2)...")
logout()  # make sure no Hugging Face auth token is used

LOCAL_MODEL_ID = "microsoft/phi-2"

tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_ID)
local_model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
).eval()

print("✅ Phi-2 loaded successfully!")


# ---------------- LLM GENERATION ----------------
def generate_llm_response(user_message, emotion="Neutral"):
    """Generate an LLM response based on text and optional emotion."""
    emotion_prompts = {
        "Neutral": "Respond naturally and conversationally.",
        "Happy": "Be enthusiastic and positive.",
        "Sad": "Show empathy and care.",
        "Angry": "Be calm and de-escalate.",
        "Fearful": "Be reassuring and comforting.",
        "Disgust": "Be honest and understanding.",
        "Surprised": "Be curious and engaged.",
    }

    system_prompt = (
        f"You are an empathetic AI assistant. "
        f"The user seems {emotion.lower()}. "
        f"{emotion_prompts.get(emotion, emotion_prompts['Neutral'])}\n"
        f"---\n"
        f"User: {user_message}\n"
        f"Assistant:"
    )

    inputs = tokenizer(system_prompt, return_tensors="pt").to(device)
    outputs = local_model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Assistant:" in response:
        response = response.split("Assistant:")[-1].strip()

    return response


# ---------------- AUDIO PREPROCESSING ----------------
def preprocess_audio_to_spikes(audio_bytes, T=32):
    waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
    waveform = waveform.mean(dim=0, keepdim=True)

    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )(waveform)

    mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    mel_spec = mel_spec.squeeze(0).T  # [time, mel]

    feat = mel_spec.mean(dim=0).numpy()
    feat = np.pad(feat, (0, max(0, 1611 - len(feat))))[:1611]
    feat = feat.astype(np.float32)
    feat = feat / (np.linalg.norm(feat) + 1e-6)
    feat = np.clip(feat, 0, 1.0)

    S = spikegen.rate(torch.tensor(feat), num_steps=T).float()
    mask = torch.ones(T, dtype=torch.bool)
    return S.unsqueeze(0), mask.unsqueeze(0)


# ---------------- EMOTION DETECTION ----------------
def detect_emotion(audio_bytes):
    try:
        spikes, mask = preprocess_audio_to_spikes(audio_bytes, T=T)
        spikes, mask = spikes.to(device), mask.to(device)

        functional.reset_net(model)
        with torch.no_grad():
            logits = model(spikes, mask=mask)
            pred = logits.argmax(dim=1).item()
            confidence = torch.softmax(logits, dim=1)[0, pred].item()

        emotion = label_map.get(pred, "Neutral")
        return emotion, confidence
    except Exception as e:
        print(f"Error during emotion detection: {e}")
        return "Neutral", 0.0


# ---------------- ROUTES ----------------
@app.get("/")
async def health():
    return {"message": "Server Running"}


# 💬 Text input — use LLM
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    llm_response = generate_llm_response(request.message, "Neutral")
    return ChatResponse(response=llm_response, emotion="Neutral")


# 🎙️ Voice input — only emotion prediction
@app.post("/api/voice", response_model=ChatResponse)
async def voice(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    try:
        emotion, confidence = detect_emotion(audio_bytes)
        print(f"Detected emotion: {emotion} (confidence: {confidence:.2f})")

        # Return only emotion prediction, no LLM
        return ChatResponse(
            response=f"Detected emotion: {emotion} (confidence: {confidence:.2f})",
            emotion=emotion
        )
    except Exception as e:
        print(f"Error processing voice input: {e}")
        return ChatResponse(response="Error processing audio", emotion="Neutral")


# Run with:
# pip install fastapi uvicorn torchaudio snntorch transformers huggingface_hub
# uvicorn main:app --reload