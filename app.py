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
}

# ---------------- STABLE MODELS ----------------
VISION_MODEL_API = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
LLM_MODEL_API = "https://api-inference.huggingface.co/models/google/gemma-2b-it"

# ---------------- SESSION STATE ----------------
if "landmark" not in st.session_state:
    st.session_state.landmark = ""

if "chat" not in st.session_state:
    st.session_state.chat = []

# ---------------- API HELPER ----------------
def call_hf(api_url, payload=None, is_image=False):
    for _ in range(5):
        if is_image:
            response = requests.post(
                api_url,
                headers=HEADERS,
                data=payload
            )
        else:
            response = requests.post(
                api_url,
                headers={**HEADERS, "Content-Type": "application/json"},
                json=payload
            )

        if response.status_code == 200:
            return response

        if response.status_code == 503:
            time.sleep(8)
            continue

        return response

    return response

# ---------------- IMAGE CAPTION ----------------
def identify_landmark(image):
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")

    response = call_hf(
        VISION_MODEL_API,
        payload=img_bytes.getvalue(),
        is_image=True
    )

    if response.status_code == 200:
        return response.json()[0]["generated_text"]

    return "Could not identify landmark."

# ---------------- LLM RESPONSE ----------------
def get_travel_recommendation(prompt):
    payload = {
        "inputs": f"You are a helpful travel assistant.\n\nUser: {prompt}\nAssistant:",
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.6,
            "top_p": 0.9
        }
    }

    response = call_hf(LLM_MODEL_API, payload)

    if response.status_code == 200:
        output = response.json()
        if isinstance(output, list):
            return output[0]["generated_text"]
        return output.get("generated_text", "No response.")

    return "Model is busy. Please try again."

# ---------------- UI ----------------
st.title("Travel Recommendation Chatbot")
st.caption("Multimodal AI using Gemma (Hosted)")

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
            with st.spinner("Identifying landmark..."):
                landmark = identify_landmark(image)
                st.session_state.landmark = landmark
