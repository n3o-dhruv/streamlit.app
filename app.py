import streamlit as st

# PAGE CONFIG (FIRST!)
st.set_page_config(page_title="Smart Study Planner", layout="wide")

import requests
from PIL import Image
import io

# ---------- UI HEADER ----------
st.title("Smart Study Planner")
st.caption("Multimodal AI Assistant • MACS AIML")

# ---------- SECRETS ----------
if "HF_API_TOKEN" not in st.secrets:
    st.error("HF_API_TOKEN missing in Secrets")
    st.stop()

HF_TOKEN = st.secrets["HF_API_TOKEN"]

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# ---------- MODELS (STABLE AF) ----------
VISION_API = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
LLM_API = "https://api-inference.huggingface.co/models/google/gemma-2b-it"

# ---------- SESSION ----------
st.session_state.setdefault("notes", "")
st.session_state.setdefault("chat", [])

# ---------- FUNCTIONS ----------
def hf_call(url, payload=None, image=False):
    if image:
        return requests.post(url, headers=HEADERS, data=payload, timeout=60)
    return requests.post(url, headers=HEADERS, json=payload, timeout=60)

def analyze_image(img):
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    res = hf_call(VISION_API, buf.getvalue(), image=True)
    if res.status_code == 200:
        return res.json()[0]["generated_text"]
    return "Could not analyze image."

def get_study_plan(q):
    prompt = f"""
You are a study planner.
Give a clear 5-step study plan for:
{q}
"""
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.6
        }
    }

    res = hf_call(LLM_API, payload)

    if res.status_code == 200:
        return res.json()[0]["generated_text"]

    return "AI temporarily unavailable. Try again."

# ---------- LAYOUT ----------
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Upload Notes")
    file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if file:
        img = Image.open(file)
        st.image(img, width=300)
        if st.button("Analyze Notes"):
            with st.spinner("Analyzing..."):
                st.session_state.notes = analyze_image(img)
                st.success("Notes analyzed!")

with col2:
    st.subheader("Assistant")

    if st.session_state.notes:
        st.info(st.session_state.notes)

    for role, msg in st.session_state.chat:
        with st.chat_message(role):
            st.write(msg)

    if q := st.chat_input("Ask for a study strategy..."):
        st.session_state.chat.append(("user", q))
        with st.chat_message("assistant"):
            with st.spinner("Generating plan..."):
                reply = get_study_plan(q)
                st.write(reply)
                st.session_state.chat.append(("assistant", reply))
