import streamlit as st
import requests
from PIL import Image
import io

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Smart Study Planner",
    layout="wide"
)

# ================= UI HEADER =================
st.title("Smart Study Planner")
st.caption("Multimodal AI Assistant • MACS AIML")

# ================= SECRETS =================
if "HF_API_TOKEN" not in st.secrets:
    st.error("HF_API_TOKEN missing in Streamlit Secrets")
    st.stop()

HF_TOKEN = st.secrets["HF_API_TOKEN"]

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# ================= MODELS =================
# Vision model (stable)
VISION_API = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"

# Text model (MOST STABLE on HF free tier)
LLM_API = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

# ================= SESSION STATE =================
if "notes" not in st.session_state:
    st.session_state.notes = ""

if "chat" not in st.session_state:
    st.session_state.chat = []

# ================= HELPER FUNCTION =================
def hf_call(url, payload=None, image=False):
    try:
        if image:
            return requests.post(url, headers=HEADERS, data=payload, timeout=60)
        else:
            return requests.post(url, headers=HEADERS, json=payload, timeout=60)
    except requests.exceptions.RequestException:
        return None

# ================= IMAGE ANALYSIS =================
def analyze_image(img):
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")

    res = hf_call(VISION_API, buffer.getvalue(), image=True)

    if not res:
        return "❌ Network error while analyzing image."

    if res.status_code != 200:
        return "⚠️ Vision model busy. Try again."

    data = res.json()
    return data[0].get("generated_text", "⚠️ Could not extract text.")

# ================= STUDY PLAN =================
def get_study_plan(query):
    prompt = f"""
You are an expert academic study planner.
Create a clear, simple 5-step study plan.

Topic:
{query}
"""

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.5,
            "top_p": 0.9
        }
    }

    res = hf_call(LLM_API, payload)

    if not res:
        return "❌ Network error. Please try again."

    if res.status_code != 200:
        return "⚠️ AI model is loading or busy. Retry after a few seconds."

    data = res.json()

    if isinstance(data, dict) and "error" in data:
        return f"⚠️ {data['error']}"

    return data[0].get("generated_text", "⚠️ No response generated.")

# ================= LAYOUT =================
col1, col2 = st.columns([1, 2])

# ---------- LEFT COLUMN ----------
with col1:
    st.subheader("Upload Notes")
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, width=300)

        if st.button("Analyze Notes"):
            with st.spinner("Analyzing notes..."):
                st.session_state.notes = analyze_image(image)
                st.success("Notes analyzed successfully!")

# ---------- RIGHT COLUMN ----------
with col2:
    st.subheader("Assistant")

    if st.session_state.notes:
        st.info(st.session_state.notes)

    for role, msg in st.session_state.chat:
        with st.chat_message(role):
            st.write(msg)

    user_query = st.chat_input("Ask for a study strategy...")

    if user_query:
        st.session_state.chat.append(("user", user_query))

        with st.chat_message("assistant"):
            with st.spinner("Generating study plan..."):
                response = get_study_plan(user_query)
                st.write(response)
                st.session_state.chat.append(("assistant", response))
