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

groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

# ================= SESSION =================
st.session_state.setdefault("ocr_text", "")
st.session_state.setdefault("chat", [])
st.session_state.setdefault("ocr_debug", "")

# ================= IMAGE PREPROCESS =================
def preprocess_image(img: Image.Image) -> Image.Image:
    # convert to RGB (fixes weird modes)
    img = img.convert("RGB")

    # upscale (HUGE improvement for timetable-like docs)
    w, h = img.size
    scale = 3 if max(w, h) < 1400 else 2  # dynamic scaling
    img = img.resize((w * scale, h * scale))

    # grayscale
    img = img.convert("L")

    # contrast boost
    img = ImageEnhance.Contrast(img).enhance(2.5)

    # sharpness boost
    img = ImageEnhance.Sharpness(img).enhance(2.0)

    # slight denoise + sharpen
    img = img.filter(ImageFilter.MedianFilter(size=3))
    img = img.filter(ImageFilter.SHARPEN)

    return img

# ================= OCR: OCR.SPACE =================
def ocr_space(img: Image.Image):
    if "OCR_SPACE_API_KEY" not in st.secrets:
        return "", "OCR_SPACE_API_KEY missing"

    try:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        r = requests.post(
            "https://api.ocr.space/parse/image",
            files={"file": ("image.png", buf, "image/png")},
            data={
                "apikey": st.secrets["OCR_SPACE_API_KEY"],
                "language": "eng",

                # IMPORTANT params:
                "OCREngine": "2",       # better engine
                "scale": "true",       # auto upscale on server too
                "isTable": "true",
                "detectOrientation": "true",
                "filetype": "PNG",
            },
            timeout=45
        )

        data = r.json()

        if data.get("IsErroredOnProcessing"):
            return "", str(data.get("ErrorMessage"))

        parsed = data.get("ParsedResults", [])
        if not parsed:
            return "", "No ParsedResults received"

        text = parsed[0].get("ParsedText", "")
        return text, ""

    except Exception as e:
        return "", f"Exception: {e}"

# ================= OPTIONAL FALLBACK OCR (EASYOCR) =================
def try_easyocr(img: Image.Image):
    try:
        import numpy as np
        import easyocr

        reader = easyocr.Reader(["en"], gpu=False)

        # easyocr needs numpy image
        arr = np.array(img)
        result = reader.readtext(arr, detail=0, paragraph=True)

        return "\n".join(result).strip()
    except Exception as e:
        return ""

# ================= MASTER OCR =================
def try_ocr(img: Image.Image):
    pre = preprocess_image(img)

    # show processed image in UI for debugging
    st.image(pre, caption="Preprocessed image (OCR input)", use_container_width=True)

    # primary OCR: OCR.Space
    text, err = ocr_space(pre)

    if text and text.strip():
        return text.strip(), ""

    # fallback OCR: easyocr (optional)
    fallback = try_easyocr(pre)
    if fallback:
        return fallback.strip(), "OCR.Space failed, used EasyOCR fallback."

    # return error
    return "", err or "OCR failed completely."

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
                    "Otherwise answer generally. "
                    "If the context contains a timetable, create a structured plan."
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
        st.image(img, caption="Original image", use_container_width=True)

        if st.button("Analyze Image"):
            with st.spinner("Analyzing image (OCR)..."):
                text, debug = try_ocr(img)

                st.session_state.ocr_text = text
                st.session_state.ocr_debug = debug

                if text.strip():
                    st.success("✅ Text extracted from image!")
                else:
                    st.warning("❌ Could not extract text. Try clearer image / zoomed screenshot.")

                if debug:
                    st.info(f"OCR Debug: {debug}")

# ---------- RIGHT ----------
with col2:
    st.subheader("Assistant")

    if st.session_state.ocr_text:
        st.info("Extracted Text (OCR Output)")
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
