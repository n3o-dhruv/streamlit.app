import streamlit as st
import requests
from PIL import Image
import io

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Travel Recommendation Chatbot",
    layout="wide"
)

HF_TOKEN = st.secrets["HF_API_TOKEN"]

VISION_API = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
LLM_API = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"

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

    response = requests.post(
        VISION_API,
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        data=img_bytes.getvalue(),
        params={"wait_for_model": "true"},
        timeout=120
    )

    if response.status_code == 200:
        try:
            return response.json()[0]["generated_text"]
        except:
            return "Landmark detected but could not be described."

    return "Image model unavailable at the moment."

def get_travel_recommendation(prompt):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 300,
            "temperature": 0.7
        }
    }

    response = requests.post(
        LLM_API,
        headers=HEADERS,
        json=payload,
        params={"wait_for_model": "true"},
        timeout=120
    )

    if response.status_code == 200:
        try:
            return response.json()[0]["generated_text"]
        except:
            return "Response could not be parsed."

    return "Language model unavailable. Please try again."

# ---------------- UI ----------------
st.title("Travel Recommendation Chatbot")
st.caption("Transformer-based Multimodal AI using Hosted Inference Models")

col1, col2 = st.columns([1, 2])

# ---------------- LEFT PANEL ----------------
with col1:
    st.subheader("Upload Landmark Image")
    image_file = st.file_uploader(
        "Upload an image of a landmark",
        type=["jpg", "jpeg", "png"]
    )

    if image_file:
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Landmark", width=280)

        if st.button("Identify Landmark"):
            with st.spinner("Identifying landmark..."):
                st.session_state.landmark = identify_landmark(image)
                st.success("Landmark processed")

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
            with st.spinner("Generating recommendation..."):
                answer = get_travel_recommendation(context)
                st.write(answer)

        st.session_state.chat.append(("assistant", answer))

st.divider()
st.caption("MACS AIML • Multimodal Transformer Project")
