import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
from rapidocr_onnxruntime import RapidOCR

st.set_page_config(page_title="Vehicle Number Plate Detection", page_icon="🚗")


@st.cache_resource
def load_models():
    return YOLO("best.pt"), RapidOCR()


model, ocr = load_models()

st.title("🚗 Vehicle Number Plate Detection")
st.write("Upload a car photo. The AI finds the number plate and reads it.")

conf = st.slider("Confidence (how sure the AI must be)", 0.1, 0.9, 0.25, 0.05)
file = st.file_uploader("Upload a car image", type=["jpg", "jpeg", "png"])

if file:
    image = Image.open(file).convert("RGB")
    img = np.array(image)

    result = model.predict(img, conf=conf)[0]
    annotated = result.plot()[:, :, ::-1]
    st.image(annotated, caption="Detected plate", use_container_width=True)

    boxes = result.boxes.xyxy.cpu().numpy().astype(int)
    if len(boxes) == 0:
        st.warning("No plate found. Try a clearer photo or lower the confidence.")

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        crop = img[max(y1, 0):y2, max(x1, 0):x2]
        big = np.array(
            Image.fromarray(crop).resize((crop.shape[1] * 3, crop.shape[0] * 3))
        )
        ocr_result, _ = ocr(big)
        text = " ".join(r[1] for r in ocr_result) if ocr_result else "Could not read"
        st.image(crop, caption=f"Plate {i + 1}", width=300)
        st.success(f"Plate number: {text}")