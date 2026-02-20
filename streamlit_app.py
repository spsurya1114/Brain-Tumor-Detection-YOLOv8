import streamlit as st
from src.tumor_metrics import compute_tumor_metrics
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from src.heatmap import create_heatmap


# ----------- AUTO LOAD LATEST TRAINED MODEL -----------
@st.cache_resource
def load_model():
    run_dir = Path("runs/detect")

    if not run_dir.exists():
        st.error("No training runs found. Train model first.")
        return None

    candidates = sorted(
        run_dir.glob("brain_tumor_yolov8*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    if not candidates:
        st.error("No brain_tumor_yolov8 runs found.")
        return None

    weights_dir = candidates[0] / "weights"
    best = weights_dir / "best.pt"
    last = weights_dir / "last.pt"

    if best.exists():
        st.success(f"Loaded model: {best}")
        return YOLO(str(best))

    elif last.exists():
        st.warning("best.pt not found. Using last.pt")
        return YOLO(str(last))

    else:
        st.error("No weights found in latest run.")
        return None


model = load_model()

# ----------- UI -----------
st.title("🧠 Brain Tumor Detection with Explainability")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None and model is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", width="stretch")

    # Run YOLO detection
    results = model(image)

    # Plot detection image
    detection_img = results[0].plot()

    # Extract tumor details
    boxes = []
    labels = []
    confidences = []

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        xyxy = box.xyxy[0].tolist()

        label = results[0].names[cls_id]
        boxes.append(xyxy)
        labels.append(label)
        confidences.append(conf)

    # Display predictions
    if labels:
        st.subheader("Detection Results")
        for l, c, b in zip(labels, confidences, boxes):
            width, height, area, severity = compute_tumor_metrics(b)
            st.write(f"✔ {l} — Confidence: {c:.2f}")
            st.write(f"   Width: {width:.1f}px  Height: {height:.1f}px")
            st.write(f"   Area: {area:.1f}px²  Severity: {severity}")


    
    else:
        st.write("No tumor detected")

    # Convert image for heatmap
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Generate heatmap overlay
    heatmap_overlay = create_heatmap(img_cv, boxes)

    # Convert BGR → RGB for Streamlit
    heatmap_overlay = cv2.cvtColor(heatmap_overlay, cv2.COLOR_BGR2RGB)

    # Convert heatmap for download
    _, buffer = cv2.imencode(".jpg", cv2.cvtColor(heatmap_overlay, cv2.COLOR_RGB2BGR))
    heatmap_bytes = buffer.tobytes()

    # ----------- SIDE BY SIDE LAYOUT -----------
    col1, col2 = st.columns(2)

    with col1:
        st.image(detection_img, caption="YOLO Tumor Detection", width="stretch")

    with col2:
        st.image(heatmap_overlay, caption="Attention Heatmap", width="stretch")

    # ----------- DOWNLOAD BUTTON -----------
    st.download_button(
        label="⬇ Download Heatmap",
        data=heatmap_bytes,
        file_name="heatmap.jpg",
        mime="image/jpeg"
    )