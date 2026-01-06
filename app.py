import streamlit as st
import requests
from PIL import Image
import io

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Smart Study Planner", layout="wide")

# ---------------- SECRETS ----------------
try:
    HF_TOKEN = st.secrets["HF_API_TOKEN"]
except KeyError:
    st.error("❌ Missing HF_API_TOKEN in Streamlit Secrets")
    st.stop()

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# ---------------- MODELS (STABLE) ----------------
VISION_API = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
LLM_API = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"

# ---------------- SESSION STATE ----------------
if "notes_text" not in st.session_state:
    st.session_state.notes_text = ""
if "chat" not in st.session_state:
    st.session_state.chat = []

# ---------------- FUNCTIONS ----------------
def query_api(url, payload=None, is_image=False):
    params = {"wait_for_model": "true"}
    if is_image:
        return requests.post(url, headers=HEADERS, data=payload, params=params, timeout=60)
    return requests.post(url, headers=HEADERS, json=payload, params=params, timeout=60)

def extract_info(image):
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    res = query_api(VISION_API, buf.getvalue(), is_image=True)

    if res.status_code == 200:
        return res.json()[0]["generated_text"]
    return "Could not read the image."

def generate_plan(user_input):
    context = st.session_state.notes_text or "General study topics"

    prompt = f"""
You are a smart study planner.
Context: {context}
Question: {user_input}

Give 3 concise and practical study tips.
"""

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.7,
            "return_full_text": False
        }
    }
