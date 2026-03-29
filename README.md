# ♻️ SmartWaste AI – Waste Segregation System

## 🚀 Overview
SmartWaste AI is a deep learning-based waste classification system that identifies waste types from images and helps in smart waste management.

---

## 🧠 Features
- Image-based waste classification (6 categories)
- CNN model using EfficientNet
- Real-time prediction via Flask API
- Interactive UI for image upload
- MLflow for experiment tracking
- DVC for pipeline management
- Docker support for deployment

---

## 🗂️ Dataset
- TrashNet Dataset
- Categories:
  - Cardboard
  - Glass
  - Metal
  - Paper
  - Plastic
  - Trash

---

## ⚙️ Tech Stack
- Python
- TensorFlow / Keras
- Flask
- MLflow
- DVC
- Docker

---

## 📸 How It Works
1. Upload image
2. Model processes image
3. Predicts waste category
4. Displays result with confidence

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
python app/app.py
Model Format: .keras (latest Keras format)