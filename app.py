import streamlit as st
import requests
from PIL import Image
import io
from groq import Groq

st.set_page_config(page_title="Smart Study Planner", layout="wide")

st.title("Smart Study Planner")
st.caption("Multimodal AI Assistant • MACS AIML")

if "GROQ_API_KEY" not in st.secrets:
    st.error("GROQ_API_KEY missing in secrets")
    st.stop()

groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ---------------- SESSION ----------------
st.session_state.setdefault("timetable_text", "")
st.session_state.setdefault("chat", [])

# ---------------- OPTIONAL OCR ----------------
def try_ocr(img):
    if "OCR_SPACE_API_KEY" not in st.secrets:
        return ""

    try:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        r = requests.post(
            "https://api.ocr.space/parse/image",
            files={"file": buf},
            data={
                "apikey": st.secrets["OCR_SPACE_API_KEY"],
                "language": "eng",
                "isTable": True
            },
            timeout=25
        )

        data = r.json()
        if data.get("IsErroredOnProcessing"):
            return ""

        return data["ParsedResults"][0]["ParsedText"]

    except Exception:
        return ""

# ---------------- LLM ----------------
def generate_plan(timetable, query):
    completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are an academic study planner."
            },
            {
                "role": "user",
                "content": (
                    f"Timetable:\n{timetable}\n\n"
                    f"Request:\n{query}\n\n"
                    "Create a clear 5-step personalized study plan."
                )
            }
        ],
        temperature=0.4,
        max_tokens=400
    )

    return completion.choices[0].message.content

# ---------------- UI ----------------
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Upload Timetable Image (Optional)")
    file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

    if file:
        img = Image.open(file)
        st.image(img, width=300)

        if st.button("Try Auto-Extract"):
            with st.spinner("Trying OCR..."):
                text = try_ocr(img)
                if text.strip():
                    st.session_state.timetable_text = text
                    st.success("Timetable extracted automatically.")
                else:
                    st.warning("OCR failed. Please paste timetable manually.")

    st.subheader("Or Paste Timetable Text Manually")
    manual_text = st.text_area(
        "Paste timetable text here",
        height=200
    )

    if st.button("Use This Timetable"):
        st.session_state.timetable_text = manual_text

with col2:
    st.subheader("Assistant")

    if st.session_state.timetable_text:
        st.info("Timetable Being Used")
        st.code(st.session_state.timetable_text)

    for role, msg in st.session_state.chat:
        with st.chat_message(role):
            st.write(msg)

    q = st.chat_input("Ask for a study plan")

    if q and st.session_state.timetable_text:
        st.session_state.chat.append(("user", q))
        with st.chat_message("assistant"):
            with st.spinner("Generating plan..."):
                reply = generate_plan(st.session_state.timetable_text, q)
                st.write(reply)
                st.session_state.chat.append(("assistant", reply))

    elif q:
        st.warning("Provide timetable text first.")
