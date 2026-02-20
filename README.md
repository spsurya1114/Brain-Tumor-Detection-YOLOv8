# 🧠 Brain Tumor Detection using YOLOv8 with Explainable AI

## 📌 Overview

This project implements an AI-based brain tumor detection system using the YOLOv8 object detection framework.  
The model analyzes MRI scans and identifies four possible states:

- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

The system also includes explainability through heatmap visualization and tumor severity estimation using bounding box metrics.

---

## ✨ Key Features

### 🔍 Tumor Detection

- YOLOv8-based object detection on MRI scans
- Localization using bounding boxes
- Multi-class tumor classification

### 🧠 Explainable AI

- Heatmap overlay highlighting regions of model attention
- Improves interpretability and clinical trust

### 📏 Tumor Severity Estimation

- Computes tumor width, height, and area
- Classifies severity as Small, Medium, or Large

### 🌐 Interactive Streamlit UI

- Upload MRI scans in real time
- View detection results and heatmap side-by-side
- Download heatmap visualization

---

## 🏗️ Project Architecture

Brain-tumour-detection/
│
├── src/
│ ├── train.py
│ ├── predict.py
│ ├── inference.py
│ ├── heatmap.py
│ ├── tumor_metrics.py
│ └── config.py
│
├── streamlit_app.py
├── tests/
├── requirements.txt
├── Dockerfile
└── README.md

cd Brain-Tumor-Detection-YOLOv8

### 2️⃣ Create virtual environment

python -m venv venv
venv\Scripts\activate

### 3️⃣ Install dependencies

pip install -r requirements.txt

## 🚀 Running the Project

### ▶️ Run Streamlit UI

python -m streamlit run streamlit_app.py

Then open:

http://localhost:8501

## 🧪 Training the Model

python src/train.py

Training outputs will be saved in:
runs/detect/

## 🔎 Prediction via Script

python src/predict.py

## 📊 Tumor Severity Estimation

Severity is computed using bounding box area:

| Area (px²) | Severity |
| ---------- | -------- |
| < 2000     | Small    |
| 2000–8000  | Medium   |
| > 8000     | Large    |

## 🧠 Explainability Module

The heatmap module:

- Converts MRI to grayscale
- Enhances abnormal intensity regions
- Overlays attention heatmap on original MRI
- Highlights tumor areas using bounding box masking

## 📁 Dataset

The dataset is not included due to size constraints.

You can download it from:

- Kaggle Brain Tumor MRI Dataset
- Roboflow Universe Brain Tumor Dataset

After download, follow dataset restructuring instructions in the project.

## 📈 Evaluation Metrics

The model can be evaluated using:

- Precision
- Recall
- mAP
- Confusion Matrix

## 🔮 Future Work

- Tumor segmentation for pixel-level localization
- Multi-modal MRI integration (T1, T2, FLAIR)
- Grad-CAM based explainability
- Cloud deployment for remote access
- Patient history tracking

## 🧑‍💻 Author

Surya SP

## 📜 License

This project is intended for academic and research purposes.
