# SmartWaste/app/app.py

from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# load trained CNN model
model = tf.keras.models.load_model("models/smartwaste_final.keras")

# class labels — alphabetical, matching image_dataset_from_directory order
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']


# preprocess image — raw [0,255] pixels; preprocess_input is inside the model graph
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image, dtype=np.float32)   # do NOT divide by 255 — model handles it
    image = np.expand_dims(image, axis=0)        # (1, 224, 224, 3)
    return image


# home route — serves the UI
@app.route("/")
def home():
    return render_template("index.html")


# predict route (FILE UPLOAD)
@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    if file.filename == "":
        return jsonify({"error": "Empty file"}), 400

    try:
        image     = Image.open(file).convert('RGB')
        processed = preprocess_image(image)

        preds       = model.predict(processed, verbose=0)[0]   # shape: (6,)
        class_index = int(np.argmax(preds))
        confidence  = float(np.max(preds))

        # Return full probability distribution so the UI can draw all bars
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