import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import pandas as pd
import time

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="SmartVision AI", layout="wide", page_icon="🧠")

# --------------------------------------------------
# GLOBAL STYLES — navy/slate palette, CSS only
# --------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg:       #0f1117;
    --surface:  #161b27;
    --surface2: #1c2333;
    --border:   #2a3348;
    --accent:   #5b8dee;
    --accent2:  #3dd68c;
    --text:     #dce3f0;
    --muted:    #6b7a99;
    --mono:     'JetBrains Mono', monospace;
    --sans:     'Inter', sans-serif;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
}

#MainMenu, footer { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-family: var(--sans) !important; }
[data-testid="stSidebar"] .stRadio label {
    color: var(--muted) !important;
    font-size: 0.88rem !important;
    padding: 6px 10px !important;
    border-radius: 7px !important;
    transition: color 0.15s, background 0.15s !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
    color: var(--text) !important;
    background: var(--surface2) !important;
}

/* ── Typography ── */
h1, h2, h3 {
    font-family: var(--sans) !important;
    color: var(--text) !important;
    letter-spacing: -0.02em !important;
}
h1 { font-size: 1.7rem !important; font-weight: 700 !important; }
h2 { font-size: 1.2rem !important; font-weight: 600 !important; }
h3 { font-size: 1rem  !important; font-weight: 600 !important; }

hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 0.5rem 0 1.2rem 0 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: var(--sans) !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 0.55rem 1.6rem !important;
    transition: opacity 0.15s, transform 0.12s !important;
}
.stButton > button:hover {
    opacity: 0.85 !important;
    transform: translateY(-1px) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploadDropzone"] {
    background: var(--surface) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploadDropzone"]:hover { border-color: var(--accent) !important; }

/* ── Metric widget ── */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: var(--muted) !important;
}
[data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-family: var(--mono) !important;
    font-size: 1.6rem !important;
}

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div {
    background: var(--border) !important;
    border-radius: 4px !important;
    height: 6px !important;
}
[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, var(--accent), var(--accent2)) !important;
    border-radius: 4px !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] th {
    background: var(--surface2) !important;
    color: var(--muted) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
}

/* ── Image ── */
[data-testid="stImage"] img {
    border-radius: 10px !important;
    border: 1px solid var(--border) !important;
}

/* ── Alerts ── */
[data-testid="stAlert"] { border-radius: 8px !important; font-size: 0.85rem !important; }

/* ── Slider ── */
[data-testid="stSlider"] > div > div > div { background: var(--accent) !important; }

/* ── Caption ── */
.stCaption { color: var(--muted) !important; font-size: 0.78rem !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------
# CLASS NAMES
# --------------------------------------------------
CLASS_NAMES = [
    "airplane",
    "bed",
    "bench",
    "bicycle",
    "bird",
    "bottle",
    "bowl",
    "bus",
    "cake",
    "car",
    "cat",
    "chair",
    "couch",
    "cow",
    "cup",
    "dog",
    "elephant",
    "horse",
    "motorcycle",
    "person",
    "pizza",
    "potted plant",
    "stop sign",
    "traffic light",
    "train",
    "truck",
]

# --------------------------------------------------
# MODEL PATHS
# --------------------------------------------------
PATHS = {
    "VGG16": "Models/vgg16_fixed.h5",
    "MobileNetV2": "Models/mobilenetv2_fixed.h5",
    "ResNet50": "Models/resnet50_fixed.h5",
    "EfficientNet": "Models/efficientnet_fixed.keras",
    "YOLO": "smartvision_yolo/weights/best.pt",
}


# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
@st.cache_resource
def load_models():
    models = {}
    try:
        models["VGG16"] = load_model(PATHS["VGG16"], compile=False)
        models["MobileNetV2"] = load_model(PATHS["MobileNetV2"], compile=False)
        models["ResNet50"] = load_model(PATHS["ResNet50"], compile=False)
        models["EfficientNet"] = load_model(PATHS["EfficientNet"], compile=False)
    except Exception as e:
        st.error(f"CNN Load Error: {e}")
    try:
        models["YOLO"] = YOLO(PATHS["YOLO"])
    except Exception as e:
        st.error(f"YOLO Load Error: {e}")
    return models


models = load_models()


# --------------------------------------------------
# PREPROCESS
# --------------------------------------------------
def preprocess(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    return np.expand_dims(img, 0)


# --------------------------------------------------
# HELPER — convert numpy image to PNG bytes for download
# --------------------------------------------------
def image_to_bytes(img_np):
    img_pil = Image.fromarray(img_np)
    buf = st.session_state.get("_buf") or __import__("io").BytesIO()
    buf = __import__("io").BytesIO()
    img_pil.save(buf, format="PNG")
    return buf.getvalue()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown("## 🧠 SmartVision AI")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "🧠 Image Classification",
            "📦 Object Detection",
            "📊 Model Performance",
            "📷 Webcam Detection",
            "ℹ️ About",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("SYSTEM STATUS")
    st.caption(f"{'✅' if 'VGG16' in models else '❌'} CNN Models")
    st.caption(f"{'✅' if 'YOLO' in models else '❌'} YOLO Detector")
    st.caption("📦 26 object classes")

# ==================================================
# HOME PAGE
# ==================================================
if page == "🏠 Home":
    st.title("🧠 SmartVision AI")
    st.markdown("*Multi-model computer vision platform — classify, detect, compare.*")
    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    c1.metric("CNN Classifiers", "4")
    c2.metric("YOLO Detector", "1")
    c3.metric("Object Classes", "26")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### What you can do")
    st.write(
        "Upload any image to classify it across VGG16, MobileNetV2, ResNet50, and EfficientNet simultaneously. "
        "Run real-time object detection with YOLOv8 and adjust confidence thresholds. "
        "Snap photos from your webcam, or compare model accuracy and loss on the performance dashboard."
    )

# ==================================================
# CLASSIFICATION PAGE
# ==================================================
elif page == "🧠 Image Classification":
    st.title("🧠 Image Classification")
    st.markdown("*Run inference across all 4 CNN models simultaneously.*")
    st.markdown("---")

    uploaded = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        col_img, col_btn = st.columns([3, 1])
        with col_img:
            st.image(image, use_column_width=True)
        with col_btn:
            st.markdown("<br><br><br>", unsafe_allow_html=True)
            run = st.button("▶ Classify")

        if run:
            img = preprocess(image)
            summary_rows = []

            st.markdown("---")
            left_col, right_col = st.columns(2)
            col_map = {0: left_col, 1: right_col, 2: left_col, 3: right_col}
            col_idx = 0

            for name in ["VGG16", "MobileNetV2", "ResNet50", "EfficientNet"]:
                if name not in models:
                    col_idx += 1
                    continue

                start = time.time()
                pred = models[name].predict(img, verbose=0)[0]
                elapsed = round(time.time() - start, 3)

                top5_idx = np.argsort(pred)[-5:][::-1]
                top5 = [(CLASS_NAMES[i], float(pred[i])) for i in top5_idx]

                summary_rows.append(
                    {
                        "Model": name,
                        "Top Prediction": top5[0][0],
                        "Confidence": f"{top5[0][1]:.2%}",
                        "Inference (s)": elapsed,
                    }
                )

                with col_map[col_idx % 4]:
                    st.markdown(f"**{name}** — `{elapsed}s`")
                    for label, conf in top5:
                        st.write(f"{label}: `{conf:.4f}`")
                        st.progress(float(conf))
                    st.markdown(" ")

                col_idx += 1

            if summary_rows:
                st.markdown("---")
                st.markdown("#### Summary")
                st.dataframe(
                    pd.DataFrame(summary_rows),
                    use_container_width=True,
                    hide_index=True,
                )

# ==================================================
# OBJECT DETECTION PAGE
# ==================================================
elif page == "📦 Object Detection":
    st.title("📦 Object Detection")
    st.markdown("*YOLOv8 bounding box detection with adjustable confidence.*")
    st.markdown("---")

    col_up, col_conf = st.columns([3, 1])
    with col_up:
        uploaded = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    with col_conf:
        st.markdown("<br>", unsafe_allow_html=True)
        conf = st.slider("Confidence", 0.1, 1.0, 0.5, step=0.05)

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        img_np = np.array(image)

        col_orig, col_det = st.columns(2)
        with col_orig:
            st.caption("Original")
            st.image(image, use_column_width=True)

        if st.button("▶ Detect Objects"):
            with st.spinner("Running YOLO..."):
                results = models["YOLO"](img_np, conf=conf)
                annotated = results[0].plot()

            with col_det:
                st.caption("Detections")
                st.image(annotated, use_column_width=True)
                st.download_button(
                    label="⬇️ Download Annotated Image",
                    data=image_to_bytes(annotated),
                    file_name="smartvision_detection.png",
                    mime="image/png",
                    use_container_width=True,
                )

            boxes = results[0].boxes
            if len(boxes) == 0:
                st.warning("No objects detected at this confidence threshold.")
            else:
                st.markdown(f"**{len(boxes)} object(s) found**")
                for box in boxes:
                    cls = int(box.cls[0])
                    score = float(box.conf[0])
                    st.write(f"{CLASS_NAMES[cls]} : {score:.3f}")

# ==================================================
# PERFORMANCE PAGE
# ==================================================
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")
    st.markdown("*Accuracy & loss comparison across all CNN architectures.*")
    st.markdown("---")

    data = {
        "Model": ["EfficientNet", "ResNet50", "VGG16", "MobileNetV2"],
        "Accuracy": [0.4911, 0.4536, 0.3040, 0.2995],
        "Loss": [2.2908, 2.0893, 2.4367, 2.3786],
    }
    df = pd.DataFrame(data)

    best_model = df.loc[df["Accuracy"].idxmax(), "Model"]
    best_acc = df["Accuracy"].max()
    best_loss = df["Loss"].min()

    c1, c2, c3 = st.columns(3)
    c1.metric("Best Accuracy", f"{best_acc:.1%}", best_model)
    c2.metric("Lowest Loss", f"{best_loss:.4f}")
    c3.metric("Models Compared", "4")

    st.markdown("<br>", unsafe_allow_html=True)
    col_acc, col_loss = st.columns(2)
    with col_acc:
        st.caption("Accuracy by Model")
        st.bar_chart(df.set_index("Model")["Accuracy"])
    with col_loss:
        st.caption("Loss by Model")
        st.bar_chart(df.set_index("Model")["Loss"])

    st.markdown("#### Full Results")
    st.table(df)

# ==================================================
# WEBCAM PAGE
# ==================================================
elif page == "📷 Webcam Detection":
    st.title("📷 Webcam Detection")
    st.markdown("*Snap a photo and run YOLO detection instantly.*")
    st.markdown("---")

    camera = st.camera_input("Take a photo")

    if camera:
        image = Image.open(camera).convert("RGB")
        img_np = np.array(image)

        with st.spinner("Detecting..."):
            results = models["YOLO"](img_np)
            annotated = results[0].plot()

        col_raw, col_ann = st.columns(2)
        with col_raw:
            st.caption("Original")
            st.image(image, use_column_width=True)
        with col_ann:
            st.caption("Detected")
            st.image(annotated, use_column_width=True)
            st.download_button(
                label="⬇️ Download Annotated Image",
                data=image_to_bytes(annotated),
                file_name="smartvision_webcam.png",
                mime="image/png",
                use_container_width=True,
            )

        boxes = results[0].boxes
        if len(boxes):
            st.markdown(f"**{len(boxes)} object(s) found**")
            for box in boxes:
                cls = int(box.cls[0])
                score = float(box.conf[0])
                st.write(f"{CLASS_NAMES[cls]} : {score:.3f}")

# ==================================================
# ABOUT PAGE
# ==================================================
elif page == "ℹ️ About":
    st.title("ℹ️ About SmartVision AI")
    st.markdown("---")

    st.markdown("#### Stack")
    st.write(
        "**Classification** — VGG16, MobileNetV2, ResNet50, EfficientNet (TensorFlow / Keras)  \n"
        "**Detection** — YOLOv8 (Ultralytics) trained on a 26-class COCO subset  \n"
        "**Frontend** — Streamlit  \n"
        "**Image Processing** — OpenCV · Pillow · NumPy"
    )

    st.markdown("#### Dataset — 26 COCO Classes")
    st.write(", ".join(CLASS_NAMES))
