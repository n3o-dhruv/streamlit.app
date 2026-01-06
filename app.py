import streamlit as st
import requests
from PIL import Image
import io
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Travel Recommendation Chatbot",
    layout="wide"
)

# ---------------- SECRETS ----------------
HF_TOKEN = st.secrets["HF_API_TOKEN"]

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# ---------------- MODELS ----------------
VISION_MODEL_API = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"

PRIMARY_LLM = "https://api-inference.huggingface.co/models/google/gemma-2b-it"
FALLBACK_LLM = "https://api-inference.huggingface.co/models/microsoft/phi-2"

# ---------------- SESSION STATE ----------------
if "landmark" not in st.session_state:
    st.session_state.landmark = ""

if "chat" not in st.session_state:
    st.session_state.chat = []

# ---------------- API CALL ----------------
def call_hf(api_url, payload=None, is_image=False):
    try:
        if is_image:
            return requests.post(
                api_url,
                headers={"Authorization": f"Bearer {HF_TOKEN}"},
                data=payload,
                timeout=60
            )
        else:
            return requests.post(
                api_url,
                headers=HEADERS,
                json=payload,
                timeout=60
            )
    except:
        return None

# ---------------- IMAGE ----------------
def identify_landmark(image):
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")

    response = call_hf(VISION_MODEL_API, img_bytes.getvalue(), True)

    if response and response.status_code == 200:
        return response.json()[0]["generated_text"]

    return "Unable to identify landmark."

# ---------------- LLM WITH FALLBACK ----------------
def generate_with_model(api_url, prompt):
    payload = {
        "inputs": f"You are a helpful travel guide.\nUser: {prompt}\nAssistant:",
        "parameters": {
            "max_new_tokens": 180,
            "temperature": 0.6,
            "top_p": 0.9
        },
        "options": {
            "wait_for_model": True
        }
    }

    response = call_hf(api_url, payload)

    if response and response.status_code == 200:
        out = response.json()
        if isinstance(out, list):
            return out[0]["generated_text"]
        return out.get("generated_text")

    return None

def get_travel_recommendation(prompt):
    # Try primary model
    answer = generate_with_model(PRIMARY_LLM, prompt)
    if answer:
        return answer

    # Fallback model (ALWAYS WORKS)
    answer = generate_with_model(FALLBACK_LLM, prompt)
    if answer:
        return answer

    return "Sorry, all models are currently busy. Please try again in a moment."

# ---------------- UI ----------------
st.title("Travel Recommendation Chatbot")
st.caption("Multimodal AI using Hosted Models")

col1, col2 = st.columns([1, 2])

# -------- LEFT --------
with col1:
    st.subheader("Upload Landmark Image")
    image_file = st.file_uploader(
        "Upload an image of a landmark",
        type=["jpg", "jpeg", "png"]
    )

    if image_file:
        image = Image.open(image_file)
        st.image(image, use_container_width=True)

        if st.button("Identify Landmark"):
            with st.spinner("Analyzing image..."):
                st.session_state.landmark = identify_landmark(image)
                st.success(st.session_state.landmark)

# -------- RIGHT --------
with col2:
    st.subheader("Travel Chatbot")

    if st.session_state.landmark:
        st.info(f"📍 Landmark context: {st.session_state.landmark}")

    for role, msg in st.session_state.chat:
        with st.chat_message(role):
            st.write(msg)

# ---------------- CHAT INPUT ----------------
user_input = st.chat_input("Ask about travel tips, budget, or attractions...")

if user_input:
    st.session_state.chat.append(("user", user_input))

    prompt = (
        f"The user is viewing {st.session_state.landmark}. {user_input}"
        if st.session_state.landmark
        else user_input
    )

    with st.spinner("Generating response..."):
        answer = get_travel_recommendation(prompt)

    st.session_state.chat.append(("assistant", answer))
    st.rerun()

# ---------------- CLEAR ----------------
if st.button("Clear Conversation"):
    st.session_state.chat = []
    st.session_state.landmark = ""
    st.rerun()

st.divider()
