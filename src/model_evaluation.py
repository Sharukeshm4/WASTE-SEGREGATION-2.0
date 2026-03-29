# SmartWaste/src/model_evaluation.py

import numpy as np
import tensorflow as tf
import mlflow
from sklearn.metrics import classification_report, confusion_matrix

from src.data_preprocessing import preprocess


def evaluate(split_dir: str = "data/split") -> None:
    """
    Loads the final trained Keras model and evaluates it on the held-out
    test set produced by preprocess(). Logs metrics to MLflow.
    """
    _, _, test_ds, class_names = preprocess(split_dir)

    model = tf.keras.models.load_model("models/smartwaste_final.keras")

    # ── Basic metrics ──────────────────────────────────────────────────────────
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"\nTest accuracy : {test_acc:.4f}")
    print(f"Test loss     : {test_loss:.4f}")

    mlflow.log_metric("test_accuracy", test_acc)
    mlflow.log_metric("test_loss", test_loss)

    # ── Per-class report ───────────────────────────────────────────────────────
    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    evaluate()