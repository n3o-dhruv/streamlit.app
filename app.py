import streamlit as st
import requests
from PIL import Image
import io

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Travel Recommendation Chatbot",
    layout="wide"
)

# ---------------- SECRETS ----------------
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

# ---------------- SESSION STATE ----------------
if "landmark" not in st.session_state:
    st.session_state.landmark = ""

if "chat" not in st.session_state:
    st.session_state.chat = []

# ---------------- IMAGE (SIMPLE CONTEXT) ----------------
def identify_landmark(image):
    return "a famous tourist landmark"

# ---------------- OPENROUTER LLM ----------------
def get_travel_recommendation(prompt):
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://streamlit.app",
        "X-Title": "Travel Recommendation Chatbot"
    }

    payload = {
        "model": "google/gemma-2-9b-it",
        "messages": [
            {"role": "system", "content": "You are a professional travel guide."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 300,
        "temperature": 0.6
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]

    return "Unable to generate response at the moment."

# ---------------- UI ----------------
st.title("Travel Recommendation Chatbot")
st.caption("Multimodal AI using Hosted Models (OpenRouter)")

col1, col2 = st.columns([1, 2])

# -------- LEFT PANEL --------
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
            st.session_state.landmark = identify_landmark(image)
            st.success(st.session_state.landmark)

# -------- RIGHT PANEL --------
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
