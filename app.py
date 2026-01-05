import streamlit as st
import requests
from PIL import Image
import io
import time

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Travel Recommendation Chatbot",
    layout="wide"
)

HF_TOKEN = st.secrets["HF_API_TOKEN"]

VISION_MODEL_API = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
LLM_MODEL_API = "https://api-inference.huggingface.co/models/google/gemma-2b-it"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# ---------------- SESSION STATE ----------------
if "landmark" not in st.session_state:
    st.session_state.landmark = ""

if "chat" not in st.session_state:
    st.session_state.chat = []

# ---------------- FUNCTIONS ----------------
def identify_landmark(image):
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")

    for _ in range(3):
        response = requests.post(
            VISION_MODEL_API,
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            data=img_bytes.getvalue(),
            timeout=60
        )

        if response.status_code == 200:
            try:
                return response.json()[0]["generated_text"]
            except:
                return "Unable to extract landmark description."

        time.sleep(5)

    return "Vision model is busy. Please try again."

def get_travel_recommendation(prompt):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 300,
            "temperature": 0.7
        }
    }

    for _ in range(3):
        response = requests.post(
            LLM_MODEL_API,
            headers=HEADERS,
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            try:
                return response.json()[0]["generated_text"]
            except:
                return "Response could not be parsed."

        time.sleep(5)

    return "Language model is busy. Please try again."

# ---------------- UI ----------------
st.title("Travel Recommendation Chatbot")
st.caption("Transformer-based Multimodal AI using Hosted Inference Models")

col1, col2 = st.columns([1, 2])

# ---------------- LEFT PANEL ----------------
with col1:
    st.subheader("Upload Landmark Image")
    image_file = st.file_uploader(
        "Upload a landmark image",
        type=["jpg", "jpeg", "png"]
    )

    if image_file:
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Image", width=280)

        if st.button("Identify Landmark"):
            with st.spinner("Processing image..."):
                st.session_state.landmark = identify_landmark(image)
                st.success("Landmark identified")

# ---------------- RIGHT PANEL ----------------
with col2:
    st.subheader("Travel Chatbot")

    if st.session_state.landmark:
        st.markdown(f"**Identified Landmark:** {st.session_state.landmark}")

    for role, msg in st.session_state.chat:
        with st.chat_message(role):
            st.write(msg)

    user_input = st.chat_input(
        "Ask about best time to visit, budget, attractions, or travel tips"
    )

    if user_input:
        st.session_state.chat.append(("user", user_input))

        context = f"""
You are a professional travel guide.

Landmark:
{st.session_state.landmark}

Conversation:
"""

        for r, m in st.session_state.chat:
            context += f"{r}: {m}\n"

        with st.chat_message("assistant"):
