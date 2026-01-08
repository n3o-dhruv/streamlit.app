import streamlit as st
import requests
from PIL import Image
import io
from groq import Groq

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Smart Study Planner", layout="wide")

# ================= UI =================
st.title("Smart Study Planner")
st.caption("Multimodal AI Assistant • MACS AIML")

# ================= SECRETS =================
if "HF_API_TOKEN" not in st.secrets or "GROQ_API_KEY" not in st.secrets:
    st.error("API keys missing in Streamlit Secrets")
    st.stop()

HF_TOKEN = st.secrets["HF_API_TOKEN"]
GROQ_KEY = st.secrets["GROQ_API_KEY"]

# ================= CLIENTS =================
groq_client = Groq(api_key=GROQ_KEY)

VISION_API = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# ================= SESSION =================
st.session_state.setdefault("notes", "")
st.session_state.setdefault("chat", [])

# ================= IMAGE ANALYSIS =================
def analyze_image(img):
    try:
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        res = requests.post(
            VISION_API,
            headers=HEADERS,
            data=buf.getvalue(),
            timeout=60
        )

        if res.status_code != 200:
            return "⚠️ Image model busy. Try again."

        return res.json()[0]["generated_text"]

    except Exception:
        return "⚠️ Image analysis failed."

# ================= STUDY PLAN (GROQ) =================
def get_study_plan(query):
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert academic study planner."
                },
                {
                    "role": "user",
                    "content": f"Create a clear 5-step study plan for:\n{query}"
                }
            ],
            temperature=0.5,
            max_tokens=300
        )

        return completion.choices[0].message.content

    except Exception:
        return "⚠️ LLM temporarily unavailable. Retry."

# ================= LAYOUT =================
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
