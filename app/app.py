from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# 🔥 LAZY LOAD MODEL (IMPORTANT FIX)
model = None

def get_model():
    global model
    if model is None:
        model = tf.keras.models.load_model("models/smartwaste_final.keras")
    return model


CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']


def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image, dtype=np.float32)
    image = np.expand_dims(image, axis=0)
    return image


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    if file.filename == "":
        return jsonify({"error": "Empty file"}), 400

    try:
        image = Image.open(file).convert('RGB')
        processed = preprocess_image(image)

        # 🔥 LOAD MODEL ONLY WHEN NEEDED
        model = get_model()

        preds = model.predict(processed, verbose=0)[0]
        class_index = int(np.argmax(preds))
        confidence = float(np.max(preds))

        all_probs = {cls: round(float(p), 6) for cls, p in zip(CLASSES, preds)}

        return jsonify({
            "prediction": CLASSES[class_index],
            "confidence": confidence,
            "probabilities": all_probs
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)