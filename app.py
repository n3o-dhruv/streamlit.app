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
if "GROQ_API_KEY" not in st.secrets or "OCR_SPACE_API_KEY" not in st.secrets:
    st.error("API keys missing in Streamlit Secrets")
    st.stop()

GROQ_KEY = st.secrets["GROQ_API_KEY"]
OCR_KEY = st.secrets["OCR_SPACE_API_KEY"]

# ================= CLIENT =================
groq_client = Groq(api_key=GROQ_KEY)

# ================= SESSION =================
st.session_state.setdefault("ocr_text", "")
st.session_state.setdefault("chat", [])

# ================= OCR (OCR.SPACE) =================
def extract_text_from_image(img):
    try:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        response = requests.post(
            "https://api.ocr.space/parse/image",
            files={"file": buf},
            data={
                "apikey": OCR_KEY,
                "language": "eng",
                "isTable": True
            },
            timeout=60
        )

        result = response.json()

        if result.get("IsErroredOnProcessing"):
            return "Failed to read timetable from image."

        parsed_results = result.get("ParsedResults")
        if not parsed_results:
            return "No readable text found in image."

        return parsed_results[0].get("ParsedText", "")

    except Exception:
        return "OCR failed due to network error."

# ================= STUDY PLAN (GROQ) =================
def generate_study_plan(timetable_text, user_query):
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an academic study planner. "
                        "You will be given timetable text extracted from an image."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Timetable Text:\n{timetable_text}\n\n"
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
        return "Study plan generation failed. Please retry."

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
            with st.spinner("Extracting timetable text..."):
                st.session_state.ocr_text = extract_text_from_image(image)
                st.success("Timetable extracted.")

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
        st.warning("Upload and analyze a timetable image first.")
