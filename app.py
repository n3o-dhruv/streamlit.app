import streamlit as st
import requests
from PIL import Image
import io

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Smart Study Planner",
    layout="wide"
)

# ---------------- SECRETS ----------------
HF_TOKEN = st.secrets["HF_API_TOKEN"]

VISION_MODEL_API = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
LLM_MODEL_API = "https://api-inference.huggingface.co/models/google/flan-t5-base"

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# ---------------- SESSION STATE ----------------
if "notes_text" not in st.session_state:
    st.session_state.notes_text = ""

if "chat" not in st.session_state:
    st.session_state.chat = []

# ---------------- FUNCTIONS ----------------
def extract_notes(image):
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")

    response = requests.post(
        VISION_MODEL_API,
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        data=img_bytes.getvalue(),
        params={"wait_for_model": "true"},
        timeout=120
    )

    if response.status_code == 200:
        try:
            return response.json()[0]["generated_text"]
        except:
            return "Notes detected but could not be read."

    return "Notes image model unavailable."

def generate_study_plan(prompt):
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 120,
            "temperature": 0.6
        }
    }

    response = requests.post(
        LLM_MODEL_API,
        headers=HEADERS,
        json=payload,
        params={"wait_for_model": "true"},
        timeout=120
    )

    if response.status_code == 200:
        try:
            return response.json()[0]["generated_text"]
        except:
            return "Could not generate study plan."

    return "Study planner model unavailable."

# ---------------- UI ----------------
st.title("Smart Study Planner")
st.caption("Transformer-based Multimodal AI using Hosted Inference Models")

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
            with st.spinner("Extracting notes..."):
                st.session_state.notes_text = extract_notes(image)
                st.success("Notes extracted")

# ---------------- RIGHT PANEL ----------------
with col2:
    st.subheader("Study Planner Assistant")

    if st.session_state.notes_text:
        st.markdown(f"**Extracted Notes:** {st.session_state.notes_text}")

    for role, msg in st.session_state.chat:
        with st.chat_message(role):
            st.write(msg)

    user_input = st.chat_input(
        "Ask for study plan, goals, revision strategy, or exam preparation"
    )

    if user_input:
        st.session_state.chat.append(("user", user_input))

        prompt = f"""
You are an intelligent academic study planner.

Notes:
{st.session_state.notes_text}

Student Goal:
{user_input}

Create a concise study plan with goals and revision tips.
"""

        with st.chat_message("assistant"):
            with st.spinner("Generating study plan..."):
                answer = generate_study_plan(prompt)
                st.write(answer)

        st.session_state.chat.append(("assistant", answer))

st.divider()
st.caption("MACS AIML • Multimodal Transformer Project")
