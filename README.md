Python • Deep Learning • TensorFlow/PyTorch • CNN Architectures • Transfer Learning • VGG16 • ResNet50 • MobileNet • EfficientNet • Object Detection • YOLO • Computer Vision • OpenCV • Data Preprocessing • Model Evaluation • Streamlit • Hugging Face • Cloud Deployment • Image Classification • Data Visualization


# 🧠 SmartVision AI

### Intelligent Multi-Model Image Classification & Object Detection Platform

SmartVision AI is a deep-learning powered computer vision system that combines multiple CNN architectures with YOLO object detection into a single interactive web application.

It allows users to upload images, compare predictions across models, detect objects in real-time, and analyze model performance — all from one interface.

---

## 🚀 Live Demo

👉 Hugging Face Space:
https://huggingface.co/spaces/cantrachme/smartvision-ai

---

## ✨ Features

### 🧠 Multi-Model Image Classification

Runs inference across **four CNN architectures simultaneously**:

* VGG16
* MobileNetV2
* ResNet50
* EfficientNet

Shows:

* Top-5 predictions per model
* Confidence scores
* Inference time comparison

---

### 📦 Object Detection with YOLOv8

* Bounding box detection
* Adjustable confidence threshold
* Annotated image download
* Webcam detection support

Trained on **26 COCO object classes**.

---

### 📊 Model Performance Dashboard

Visual comparison of:

* Accuracy per model
* Loss per model
* Best performing architecture

---

### 📷 Webcam Detection

Capture an image directly from the browser and run YOLO detection instantly.

---

## 🏗️ Architecture

The project uses a production-style ML deployment structure:

```
User → Streamlit App (HF Space)
            ↓
      Loads models from
            ↓
     Hugging Face Model Repo
```

### Repositories

| Component     | Purpose                       |
| ------------- | ----------------------------- |
| GitHub Repo   | Source code & training assets |
| HF Model Repo | Stored trained weights        |
| HF Space      | Deployed application          |

This separation keeps the app lightweight and scalable.

---

## 🛠️ Tech Stack

**Frontend**

* Streamlit

**Deep Learning**

* TensorFlow / Keras
* Ultralytics YOLOv8

**Image Processing**

* OpenCV
* Pillow
* NumPy

**Deployment**

* Hugging Face Spaces
* Hugging Face Model Hub

---

## 📂 Project Structure

```
SmartVision-AI/
│
├── app.py
├── Models/                # Locally trained models (ignored in repo)
├── smartvision_yolo/      # YOLO training outputs
├── requirements.txt
├── README.md
└── .gitignore
└── Notebooks 
```

---

## 🧠 Dataset

The models are trained on a subset of the COCO dataset including 26 classes such as:

airplane, car, person, dog, bus, truck, chair, pizza, traffic light, etc.

---

## ⚙️ Installation (Local Run)

```bash
git clone https://github.com/cantrachme/SmartVision-AI---Intelligent-Multi-Class-Object-Recognition-System
cd SmartVision-AI
pip install -r requirements.txt
streamlit run app.py
```

---

## ☁️ Deployment Notes

* Models are hosted on Hugging Face Model Hub
* App is deployed on Hugging Face Spaces
* Large files are excluded from GitHub using `.gitignore`

This mirrors real-world ML deployment pipelines.

---

## 📌 Future Improvements

* Add video detection support
* Model ensemble voting system
* Performance benchmarking on GPU vs CPU
* Mobile-friendly UI
* Dataset expansion beyond COCO subset

---


## ⭐ Why This Project Matters

This project demonstrates:

* Multi-model deep learning pipelines
* Real-time inference deployment
* Production-ready ML architecture
* Model hosting & separation strategy
* End-to-end AI system design

It showcases both **machine learning expertise** and **deployment engineering skills**.

---
