import streamlit as st

# ✅ MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Smart Study Planner",
    layout="wide"
)

import requests
from PIL import Image
import io

# ---------------- BASIC UI (RENDER FIRST) ----------------
st.title("Smart Study Planner")
st.caption("Multimodal AI Assistant • MACS AIML")

# ---------------- SECRETS ----------------
if "HF_API_TOKEN" not in st.secrets:
    st.error("❌ HF_API_TOKEN missing in Streamlit Secrets")
    st.stop()

HF_TOKEN = st.secrets["HF_API_TOKEN"]

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# ---------------- MODELS (STABLE) ----------------
VISION_API = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
LLM_API = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"

# ---------------- SESSION STATE ----------------
st.session_state.setdefault("notes_text", "")
st.session_state.setdefault("chat", [])

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
    return "Could not read image."

def generate_plan(q):
    prompt = f"Give 3 concise study tips for: {q}"
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 200}
    }
    res = query_api(LLM_API, payload)
    if res.status_code == 200:
        return res.json()[0]["generated_text"]
    return "⏳ Model loading. Try again in 10 seconds."

# ---------------- LAYOUT ----------------
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Upload Notes")
    file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if file:
        img = Image.open(file)
        st.image(img, width=300)
        if st.button("Analyze Notes"):
            with st.spinner("Analyzing..."):
                st.session_state.notes_text = extract_info(img)
                st.success("Notes analyzed!")

with col2:
    st.subheader("Assistant")

    if st.session_state.notes_text:
        st.info(st.session_state.notes_text)

    for r, m in st.session_state.chat:
        with st.chat_message(r):
            st.write(m)

    if q := st.chat_input("Ask for a study strategy..."):
        st.session_state.chat.append(("user", q))
        with st.chat_message("assistant"):
            reply = generate_plan(q)
            st.write(reply)
            st.session_state.chat.append(("assistant", reply))
