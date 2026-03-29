# SmartWaste – Model Accuracy Improvement Guide

## 1. Fix Inference Preprocessing — `app/app.py`

**Problem:** The model has `preprocess_input` baked inside its graph (see `src/model_training.py` line 49).
Dividing by `255.0` before calling `model.predict()` corrupts the input.

```python
# ❌ OLD — double-scales pixels
image = np.array(image) / 255.0

# ✅ FIXED — pass raw [0, 255] pixels
image = np.array(image, dtype=np.float32)
```

> **Action:** Restart Flask after saving. This is the single biggest accuracy fix.

---

## 2. Add More Training Data — `data/raw/<class>/`

EfficientNetB0 fine-tuning needs at least **200–500 images per class** to generalise well.

| Class | Minimum recommended |
|---|---|
| `cardboard` | 400+ |
| `glass` | 400+ |
| `metal` | 400+ |
| `paper` | 400+ |
| `plastic` | 400+ |
| `trash` | 400+ |

**Sources:** [TrashNet dataset](https://github.com/garythung/trashnet), [Kaggle Garbage Classification](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification).

After adding images, re-run: `python -m src.data_preprocessing` then `python -m src.model_training`.

---

## 3. Tune Augmentation — `src/data_preprocessing.py` → `_augmentation_layer()`

The current augmentation is heavy. If data is limited, strong augmentation can hurt rather than help.

```python
# In _augmentation_layer() — reduce intensity if data < 200/class
tf.keras.layers.RandomRotation(0.1),   # was 0.2
tf.keras.layers.RandomZoom(0.1),       # was 0.2
tf.keras.layers.RandomBrightness(0.1), # was 0.2
```

Also consider adding `RandomAdjustSaturation` and `RandomSharpness` to help distinguish
glass vs. cardboard (texture difference).

---

## 4. Unfreeze More Layers — `src/model_training.py` → Stage 2

Waste images differ significantly from ImageNet. Try unfreezing more of the base:

```python
# Line 150 — currently unfreezes top 80 of 237 layers
for layer in base_model.layers[:-80]:
    layer.trainable = False

# Try unfreezing more — e.g. top 120 layers
for layer in base_model.layers[:-120]:
    layer.trainable = False
```

Use a **lower LR** if you unfreeze more layers (e.g. `5e-6` instead of `1e-5`).

---

## 5. Upgrade the Backbone — `src/model_training.py` → `build_model()`

EfficientNetB0 is the smallest variant. If accuracy is still low, upgrade:

```python
# Current
base_model = tf.keras.applications.EfficientNetB0(...)

# Better options (larger = more accurate, slower to train)
base_model = tf.keras.applications.EfficientNetB3(input_shape=(300, 300, 3), ...)
base_model = tf.keras.applications.EfficientNetV2S(input_shape=(384, 384, 3), ...)
```

If you change `input_shape`, also update `IMG_SIZE` in `src/data_preprocessing.py`:
```python
IMG_SIZE = (300, 300)  # match EfficientNetB3
```

---

## 6. Tune the Classification Head — `src/model_training.py` → `build_model()`

The current head is a single Dense(256). For 6 classes, try:

```python
x = layers.Dense(512, activation="relu")(x)   # wider
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(256, activation="relu")(x)   # second dense layer
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
```

---

## 7. Use Label Smoothing — `src/model_training.py` → `model.compile()`

Label smoothing prevents overconfidence and improves generalisation:

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(label_smoothing=0.1),
    metrics=["accuracy"],
)
```

---

## 8. Evaluate Properly — `src/model_evaluation.py`

After retraining, run the updated evaluation script to see **per-class accuracy**:

```bash
python -m src.model_evaluation
```

The classification report will show which classes are being confused
(e.g. `cardboard` vs `glass`) so you can target data collection.

---

## 9. Track Experiments — MLflow

All training runs are already tracked via MLflow (`src/model_training.py`). View the dashboard:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Open `http://localhost:5000` and compare stage 1 vs stage 2 metrics to diagnose where accuracy drops.

---

## Priority Order

| Priority | File | Change |
|---|---|---|
| 🔴 Critical | `app/app.py` | Remove `/255.0`, restart server |
| 🔴 Critical | `data/raw/` | More images per class |
| 🟡 High | `src/model_training.py` | Unfreeze more layers, label smoothing |
| 🟡 High | `src/data_preprocessing.py` | Tune augmentation intensity |
| 🟢 Optional | `src/model_training.py` | Upgrade to EfficientNetB3/V2 |
