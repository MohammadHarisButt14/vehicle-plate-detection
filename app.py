import io
import time

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from rapidocr_onnxruntime import RapidOCR

st.set_page_config(page_title="Plate Detector AI", page_icon="🚗", layout="wide")

# ---------- Style ----------
st.markdown(
    """
<style>
.hero {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #6a11cb 100%);
    padding: 2.2rem 2rem; border-radius: 18px; text-align: center;
    margin-bottom: 1.2rem; box-shadow: 0 8px 24px rgba(0,0,0,0.35);
}
.hero h1 { color: white; margin: 0; font-size: 2.4rem; }
.hero p { color: #dbe6ff; margin: 0.5rem 0 0 0; font-size: 1.05rem; }
.card {
    background: #161b26; border: 1px solid #2c3446; border-radius: 14px;
    padding: 1.1rem 1.2rem; height: 100%;
}
.card h4 { margin: 0 0 0.4rem 0; color: #8ab4ff; }
.plate-box {
    background: #fff7cc; color: #111; border: 3px solid #222; border-radius: 10px;
    text-align: center; font-size: 2rem; font-weight: 800; letter-spacing: 4px;
    padding: 0.6rem 1rem; margin: 0.4rem 0 0.8rem 0; font-family: monospace;
}
.footer { text-align: center; color: #8a93a6; margin-top: 2rem; font-size: 0.9rem; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------- Models ----------
@st.cache_resource(show_spinner=False)
def load_models():
    return YOLO("best.pt"), RapidOCR()


with st.spinner("Waking up the AI models..."):
    model, ocr = load_models()


def clean_text(text):
    return " ".join(text.upper().split())


# ---------- Sidebar ----------
with st.sidebar:
    st.title("⚙️ Settings")
    conf = st.slider("Confidence threshold", 0.10, 0.90, 0.25, 0.05,
                     help="How sure the AI must be before it shows a plate.")
    upscale = st.checkbox("Enlarge plate before reading (better OCR)", value=True)
    st.divider()
    st.subheader("📌 Project")
    st.write("**Model:** YOLOv8n")
    st.write("**OCR:** RapidOCR")
    st.write("**Dataset:** Roboflow License Plate Recognition")
    st.write("**Framework:** Streamlit")
    st.link_button("🔗 GitHub Repository",
                   "https://github.com/MohammadHarisButt14/vehicle-plate-detection")
    st.divider()
    st.caption("Deep Learning Semester Project")

# ---------- Hero ----------
st.markdown(
    """
<div class="hero">
  <h1>🚗 Vehicle Number Plate Detection</h1>
  <p>Upload a car photo. The AI finds the number plate and reads it in seconds.</p>
</div>
""",
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["🔍 Detect Plate", "📊 Model Performance", "ℹ️ About Project"])

# ---------- Tab 1: Detect ----------
with tab1:
    file = st.file_uploader("📤 Upload a car image (JPG, JPEG or PNG)",
                            type=["jpg", "jpeg", "png"])

    if not file:
        c1, c2, c3 = st.columns(3)
        c1.markdown('<div class="card"><h4>1️⃣ Upload</h4>Choose a clear photo of a car where the number plate is visible.</div>', unsafe_allow_html=True)
        c2.markdown('<div class="card"><h4>2️⃣ Detect</h4>YOLOv8 draws a box around the number plate.</div>', unsafe_allow_html=True)
        c3.markdown('<div class="card"><h4>3️⃣ Read</h4>OCR reads the letters and numbers from the plate.</div>', unsafe_allow_html=True)
        st.info("👆 Upload an image above to get started.")
    else:
        image = Image.open(file).convert("RGB")
        img = np.array(image)

        bar = st.progress(0, text="📥 Loading image...")
        time.sleep(0.3)

        bar.progress(25, text="🔍 Searching for number plates...")
        t0 = time.time()
        result = model.predict(img, conf=conf, verbose=False)[0]
        infer_ms = (time.time() - t0) * 1000

        bar.progress(55, text="🎨 Drawing detection boxes...")
        annotated = result.plot()[:, :, ::-1]
        time.sleep(0.3)

        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        confs = result.boxes.conf.cpu().numpy()

        bar.progress(75, text="🔤 Reading plate text...")
        plates = []
        h, w = img.shape[:2]
        for (x1, y1, x2, y2), c in zip(boxes, confs):
            x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            read_img = crop
            if upscale:
                read_img = np.array(
                    Image.fromarray(crop).resize((crop.shape[1] * 3, crop.shape[0] * 3))
                )
            ocr_result, _ = ocr(read_img)
            text = clean_text(" ".join(r[1] for r in ocr_result)) if ocr_result else "Could not read"
            plates.append((crop, text, float(c)))

        bar.progress(100, text="✅ Done!")
        time.sleep(0.4)
        bar.empty()

        # Metrics row
        m1, m2, m3 = st.columns(3)
        m1.metric("🚘 Plates found", len(plates))
        avg = (sum(p[2] for p in plates) / len(plates) * 100) if plates else 0
        m2.metric("🎯 Avg confidence", f"{avg:.1f}%")
        m3.metric("⚡ Detection time", f"{infer_ms:.0f} ms")

        st.divider()

        left, right = st.columns(2)
        with left:
            st.subheader("📷 Original")
            st.image(image, use_container_width=True)
        with right:
            st.subheader("🟩 Detected")
            st.image(annotated, use_container_width=True)

        if not plates:
            st.warning("No plate found. Try a clearer photo or lower the confidence in the sidebar.")
        else:
            st.balloons()
            st.subheader("🧾 Results")
            for i, (crop, text, c) in enumerate(plates, start=1):
                a, b = st.columns([1, 2])
                with a:
                    st.image(crop, caption=f"Plate {i} (cropped)", use_container_width=True)
                with b:
                    st.markdown(f'<div class="plate-box">{text}</div>', unsafe_allow_html=True)
                    st.progress(min(max(c, 0.0), 1.0), text=f"Detection confidence: {c * 100:.1f}%")

            buf = io.BytesIO()
            Image.fromarray(annotated).save(buf, format="PNG")
            st.download_button("⬇️ Download annotated image", data=buf.getvalue(),
                               file_name="detected_plate.png", mime="image/png")

# ---------- Tab 2: Performance ----------
with tab2:
    st.subheader("📊 Validation results")
    st.caption("Measured on 2,048 validation images after training on a 2% sample of the dataset.")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Precision", "0.93")
    k2.metric("Recall", "0.71")
    k3.metric("mAP50", "0.84")
    k4.metric("mAP50-95", "0.47")

    df = pd.DataFrame(
        {"Score": [0.93, 0.71, 0.84, 0.47]},
        index=["Precision", "Recall", "mAP50", "mAP50-95"],
    )
    st.bar_chart(df)

    st.markdown("**What these mean in easy words**")
    st.markdown(
        "- **Precision:** when the AI says 'plate', it is right about 93% of the time.\n"
        "- **Recall:** it finds about 71% of all plates that exist in the pictures.\n"
        "- **mAP50:** overall detection score, 0.84 out of 1.\n"
        "- **mAP50-95:** a stricter score that also checks how exact the box is."
    )

# ---------- Tab 3: About ----------
with tab3:
    a, b = st.columns(2)
    with a:
        st.markdown(
            """
<div class="card"><h4>🎯 Problem</h4>
Reading number plates by hand is slow. This prototype finds a plate in a photo
and reads its text automatically.</div>
""",
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(
            """
<div class="card"><h4>🗂️ Dataset</h4>
License Plate Recognition (v13) from Roboflow Universe, license CC BY 4.0.
About 98,800 training images. For this prototype only a 2% sample (about 2,000 images) was used.</div>
""",
            unsafe_allow_html=True,
        )
    with b:
        st.markdown(
            """
<div class="card"><h4>🧠 Pipeline</h4>
1. <b>YOLOv8n</b> finds the plate in the image.<br>
2. The plate is <b>cropped</b> out.<br>
3. <b>RapidOCR</b> reads the characters.<br>
4. <b>Streamlit</b> shows everything.</div>
""",
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(
            """
<div class="card"><h4>⚠️ Limitations</h4>
Trained on a small sample, so blurry, dark or very tilted plates can fail.
OCR may add or miss a character.</div>
""",
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="footer">Made with ❤️ using YOLOv8, RapidOCR and Streamlit · Deep Learning Semester Project</div>',
    unsafe_allow_html=True,
)
