# 🌐 SmartWaste AI – API Usage Guide

## 📌 Endpoint

POST /predict

---

## 📤 Request (Using cURL)

```bash
curl -X POST http://localhost:5000/predict \
-F "file=@data/raw/cardboard/cardboard1.jpg"