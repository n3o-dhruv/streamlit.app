import streamlit as st
import requests
from PIL import Image
import io
import time

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Smart Study Planner",
    layout="wide"
)

HF_TOKEN = st.secrets["HF_API_TOKEN"]

VISION_API = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
LLM_API = "https://api-inference.huggingface.co/models/google/gemma-2b-it"

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# ---------------- SESSION STATE ----------------
if "notes_summary" not in st.session_state:
    st.session_state.notes_summary = ""

if "chat" not in st.session_state:
    st.session_state.chat = []

# ---------------- FUNCTIONS ----------------
def analyze_notes(image):
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")

    for _ in range(3):
        response = requests.post(
            VISION_API,
            headers=HEADERS,
            data=img_bytes.getvalue(),
            timeout=60
        )

        if response.status_code == 200:
            try:
                return response.json()[0]["generated_text"]
            except:
                return "Notes detected but could not extract text."

        time.sleep(5)

    return "Notes image model is currently busy. Please try again."

def generate_study_plan(prompt):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 350,
            "temperature": 0.7
        }
    }

    for _ in range(3):
        response = requests.post(
            LLM_API,
            headers=HEADERS,
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            try:
                return response.json()[0]["generated_text"]
            except:
                return "Response generated but could not be parsed."

        time.sleep(5)

    return "Study planner model is busy. Please try again later."

# ---------------- UI ----------------
st.title("Smart Study Planner")
st.caption("Transformer-based Multimodal AI (Vision + Language)")

col1, col2 = st.columns([1, 2])

# ---------------- LEFT PANEL ----------------
with col1:
    st.subheader("Upload Notes Image")
    image_file = st.file_uploader(
        "Upload handwritten or printed notes",
        type=["jpg", "jpeg", "png"]
    )

    if image_file:
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Notes", width=280)

        if st.button("Analyze Notes"):
            with st.spinner("Analyzing notes..."):
                st.session_state.notes_summary = analyze_notes(image)
                st.success("Notes analyzed")

# ---------------- RIGHT PANEL ----------------
with col2:
    st.subheader("Study Planner Assistant")

    if st.session_state.notes_summary:
        st.markdown(f"**Extracted Notes Summary:** {st.session_state.notes_summary}")

    for role, msg in st.session_state.chat:
        with st.chat_message(role):
            st.write(msg)

    user_input = st.chat_input(
        "Ask for study plan, goals, revision schedule, or exam strategy"
    )

    if user_input:
        st.session_state.chat.append(("user", user_input))

        context = f"""
You are an intelligent academic study planner.

Notes Summary:
{st.session_state.notes_summary}

Conversation:
"""

        for r, m in st.session_state.chat:
            context += f"{r}: {m}\n"

        with st.chat_message("assistant"):
            with st.spinner("Generating study plan..."):
                answer = generate_study_plan(context)
                st.write(answer)

        st.session_state.chat.append(("assistant", answer))

st.divider()
st.caption("MACS AIML • Multimodal Transformer Project")
