import streamlit as st
import requests
from PIL import Image, ImageEnhance, ImageFilter
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

if "OCR_SPACE_API_KEY" not in st.secrets:
    st.error("OCR_SPACE_API_KEY missing in secrets")
    st.stop()

groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ================= SESSION =================
st.session_state.setdefault("ocr_text", "")
st.session_state.setdefault("chat", [])

# ================= IMAGE PREPROCESS =================
def preprocess_image(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size

    # upscale for small text docs
    scale = 3 if max(w, h) < 1600 else 2
    img = img.resize((w * scale, h * scale))

    # grayscale + contrast + sharpen
    img = img.convert("L")
    img = ImageEnhance.Contrast(img).enhance(2.5)
    img = ImageEnhance.Sharpness(img).enhance(2.0)

    # reduce noise slightly
    img = img.filter(ImageFilter.MedianFilter(size=3))
    img = img.filter(ImageFilter.SHARPEN)

    return img

# ================= OCR =================
def try_ocr(img: Image.Image) -> str:
    try:
        img = preprocess_image(img)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        r = requests.post(
            "https://api.ocr.space/parse/image",
            files={"file": ("image.png", buf, "image/png")},
            data={
                "apikey": st.secrets["OCR_SPACE_API_KEY"],
                "language": "eng",
                "OCREngine": "2",
                "scale": "true",
                "detectOrientation": "true",
                "isTable": "true"
            },
            timeout=60
        )

        data = r.json()

        if data.get("IsErroredOnProcessing"):
            st.error(f"OCR Error: {data.get('ErrorMessage')}")
            return ""

        parsed = data.get("ParsedResults", [])
        if not parsed:
            st.error("OCR returned no ParsedResults")
            return ""

        return parsed[0].get("ParsedText", "").strip()

    except Exception as e:
        st.error(f"OCR Exception: {e}")
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
                    "If timetable text is provided, use it to answer. "
                    "If no timetable is available, answer normally."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nUser Query:\n{query}"
            }
        ],
        temperature=0.4,
        max_tokens=600
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

        st.image(img, caption="Original image", width=350)

        if st.button("Analyze Image"):
            with st.spinner("Extracting text (OCR.Space)..."):
                text = try_ocr(img)

            if text:
                st.session_state.ocr_text = text
                st.success("✅ OCR Done! Text extracted.")
            else:
                st.warning("❌ OCR failed. Upload clearer / zoomed image.")

# ---------- RIGHT ----------
with col2:
    st.subheader("Assistant")

    if st.session_state.ocr_text:
        st.info("Extracted OCR Text")
        st.code(st.session_state.ocr_text)

    for role, msg in st.session_state.chat:
        with st.chat_message(role):
            st.write(msg)

    q = st.chat_input("Ask anything (study plans, concepts, doubts)")

    if q:
        st.session_state.chat.append(("user", q))
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                reply = generate_response(st.session_state.ocr_text, q)
                st.write(reply)
                st.session_state.chat.append(("assistant", reply))
