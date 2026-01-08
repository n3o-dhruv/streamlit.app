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

# OCR model for timetable text extraction
VISION_API = "https://api-inference.huggingface.co/models/microsoft/trocr-base-printed"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

# ================= SESSION =================
st.session_state.setdefault("ocr_text", "")
st.session_state.setdefault("chat", [])

# ================= OCR FUNCTION =================
def extract_text_from_image(img):
    try:
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        response = requests.post(
            VISION_API,
            headers=HEADERS,
            data=buffer.getvalue(),
            timeout=60
        )

        if response.status_code != 200:
            return "OCR model is currently busy. Please try again."

        data = response.json()
        return data.get("generated_text", "No readable text found in image.")

    except Exception:
        return "Failed to extract text due to a network issue."

# ================= STUDY PLAN (GROQ) =================
def generate_study_plan(timetable_text, user_query):
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an academic planner. "
                        "You are given OCR-extracted timetable text. "
                        "Use it to create a realistic and structured study plan."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"OCR Timetable Text:\n{timetable_text}\n\n"
                        f"User Request:\n{user_query}\n\n"
                        "Create a clear 5-step personalized study plan."
                    )
                }
            ],
            temperature=0.4,
            max_tokens=400
        )

        return completion.choices[0].message.content

    except Exception:
        return "The language model is temporarily unavailable. Please retry."

# ================= LAYOUT =================
col1, col2 = st.columns([1, 2])

# ---------- LEFT ----------
with col1:
    st.subheader("Upload Timetable Image")

    uploaded_file = st.file_uploader(
        "Upload timetable image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, width=300)

        if st.button("Analyze Timetable"):
            with st.spinner("Extracting text from image..."):
                st.session_state.ocr_text = extract_text_from_image(image)
                st.success("Timetable processed successfully.")

# ---------- RIGHT ----------
with col2:
    st.subheader("Assistant")

    if st.session_state.ocr_text:
        st.info("Extracted Timetable Text")
        st.code(st.session_state.ocr_text)

    for role, msg in st.session_state.chat:
        with st.chat_message(role):
            st.write(msg)

    user_query = st.chat_input("Ask for a study plan")

    if user_query and st.session_state.ocr_text:
        st.session_state.chat.append(("user", user_query))

        with st.chat_message("assistant"):
            with st.spinner("Generating study plan..."):
                reply = generate_study_plan(
                    st.session_state.ocr_text,
                    user_query
                )
                st.write(reply)
                st.session_state.chat.append(("assistant", reply))

    elif user_query:
        st.warning("Please upload and analyze a timetable image first.")
