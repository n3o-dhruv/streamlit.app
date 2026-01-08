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
if "GROQ_API_KEY" not in st.secrets:
    st.error("GROQ_API_KEY missing in secrets")
    st.stop()

groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ================= SESSION =================
st.session_state.setdefault("ocr_text", "")
st.session_state.setdefault("chat", [])

# ================= OPTIONAL OCR =================
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
            timeout=20
        )

        data = r.json()
        if data.get("IsErroredOnProcessing"):
            return ""

        return data["ParsedResults"][0]["ParsedText"]

    except Exception:
        return ""

# ================= LLM =================
def generate_response(context, query):
    completion = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful academic assistant. "
                    "If timetable text is provided, use it. "
                    "Otherwise answer generally."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Context (optional timetable text):\n{context}\n\n"
                    f"User Query:\n{query}"
                )
            }
        ],
        temperature=0.5,
        max_tokens=400
    )

    return completion.choices[0].message.content

# ================= LAYOUT =================
col1, col2 = st.columns([1, 2])

# ---------- LEFT ----------
with col1:
    st.subheader("Upload Image (Optional)")
    file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

    if file:
        img = Image.open(file)
        st.image(img, width=300)

        if st.button("Analyze Image"):
            with st.spinner("Analyzing image..."):
                text = try_ocr(img)
                if text.strip():
                    st.session_state.ocr_text = text
                    st.success("Text extracted from image.")
                else:
                    st.warning("Could not extract text. You can still chat normally.")

# ---------- RIGHT ----------
with col2:
    st.subheader("Assistant")

    if st.session_state.ocr_text:
        st.info("Extracted Text (if any)")
        st.code(st.session_state.ocr_text)

    for role, msg in st.session_state.chat:
        with st.chat_message(role):
            st.write(msg)

    q = st.chat_input("Ask anything (study plans, concepts, doubts)")

    if q:
        st.session_state.chat.append(("user", q))
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                reply = generate_response(
                    st.session_state.ocr_text,
                    q
                )
                st.write(reply)
                st.session_state.chat.append(("assistant", reply))
