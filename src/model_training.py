# SmartWaste/src/train.py

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
import mlflow

from src.data_preprocessing import (
    split_dataset,
    preprocess,
    compute_class_weights,
    _augmentation_layer,
)


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(num_classes: int, augmentation: tf.keras.Sequential) -> tuple[Model, Model]:
    """
    Functional API — freeze/unfreeze behaviour is explicit and reliable.
    Sequential wrapping of a pretrained base makes layer.trainable
    propagation ambiguous and is avoided here.

    Preprocessing pipeline inside the graph:
        raw [0,255] pixels
        → augmentation   (RandomFlip, Rotation… — no-op at inference)
        → preprocess_input  (scales to [-1, 1] as EfficientNet expects)
        → EfficientNetB0 base
        → classification head

    Because preprocessing lives inside the graph, model.predict() and
    predict_image() both receive raw pixels with no manual scaling needed.
    """
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False                # frozen for Stage 1

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = augmentation(inputs)                                               # augment
    x = tf.keras.applications.efficientnet.preprocess_input(x)            # [-1, 1]
    x = base_model(x, training=False)                                      # BN inference mode
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    return model, base_model


# ── Callbacks ─────────────────────────────────────────────────────────────────

def make_callbacks(stage: str) -> list:
    """
    Both EarlyStopping and ModelCheckpoint monitor val_accuracy so they
    agree on which epoch is 'best'. ReduceLROnPlateau monitors val_loss
    independently — it rescues plateaus before early stopping fires.
    """
    os.makedirs("models", exist_ok=True)
    return [
        EarlyStopping(
            monitor="val_accuracy",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=f"models/best_{stage}.keras",
            monitor="val_accuracy",         # same metric as EarlyStopping
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    raw_dir:   str = "data/raw",
    split_dir: str = "data/split",
):
    # Step 1 — prepare data
    split_dataset(raw_dir, split_dir)
    train_ds, val_ds, test_ds, class_names = preprocess(split_dir)

    # Compute class weights BEFORE prefetch so unbatch() is fast
    # (preprocess() already prefetches internally — weights are extracted
    #  from the non-prefetched copy inside compute_class_weights via unbatch)
    class_weights = compute_class_weights(train_ds)

    augmentation = _augmentation_layer()
    model, base_model = build_model(num_classes=len(class_names), augmentation=augmentation)

    # ── Stage 1 — head only, base frozen ─────────────────────────────────────
    #
    # lr=1e-4: lower than typical fine-tuning because the backbone is fixed —
    # only the 2-layer head is updating. 1e-3 overshoots against frozen features.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("\nStage 1 — training classification head (base fully frozen)")
    print(f"  Trainable params: {sum(tf.size(w).numpy() for w in model.trainable_weights):,}")

    with mlflow.start_run(run_name="stage1_head"):
        history1 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=25,                      # EarlyStopping(patience=7) caps this
            class_weight=class_weights,
            callbacks=make_callbacks("stage1"),
        )
        mlflow.log_params({"stage": 1, "lr": 1e-4, "layers_unfrozen": 0})
        mlflow.log_metric("best_val_accuracy", max(history1.history["val_accuracy"]))
        mlflow.log_metric("best_val_loss",     min(history1.history["val_loss"]))

    # ── Stage 2 — unfreeze top 80 layers, fine-tune ───────────────────────────
    #
    # Why 80 and not 30: EfficientNetB0 has ~237 layers. Unfreezing only 30
    # keeps mid-level texture features locked to ImageNet. Waste images differ
    # enough from ImageNet that those features need to adapt. 80 layers gives
    # the model access to texture-level features without destabilising the
    # low-level edge detectors in the early layers.
    #
    # lr=1e-5: an order of magnitude below Stage 1 because you are now touching
    # pretrained weights. A higher LR destroys the ImageNet features you
    # transferred and forces the model to relearn from scratch.
    base_model.trainable = True
    for layer in base_model.layers[:-80]:
        layer.trainable = False

    unfrozen = sum(1 for l in base_model.layers if l.trainable)
    print(f"\nStage 2 — fine-tuning top {unfrozen} base layers")
    print(f"  Trainable params: {sum(tf.size(w).numpy() for w in model.trainable_weights):,}")

    # Must recompile after changing trainable flags
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    with mlflow.start_run(run_name="stage2_finetune"):
        history2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=40,                      # EarlyStopping will cut short
            class_weight=class_weights,
            callbacks=make_callbacks("stage2"),
        )

        # Final evaluation — test_ds was never seen during training or
        # used for any callback decision, so this is an honest accuracy number
        test_loss, test_acc = model.evaluate(test_ds, verbose=0)
        print(f"\nTest accuracy : {test_acc:.4f}")
        print(f"Test loss     : {test_loss:.4f}")

        mlflow.log_params({
            "stage":           2,
            "lr":              1e-5,
            "layers_unfrozen": unfrozen,
        })
        mlflow.log_metric("best_val_accuracy", max(history2.history["val_accuracy"]))
        mlflow.log_metric("best_val_loss",     min(history2.history["val_loss"]))
        mlflow.log_metric("test_accuracy",     test_acc)    # report this number
        mlflow.log_metric("test_loss",         test_loss)

    # Save — .keras preserves the full model graph including the augmentation
    # layer and preprocess_input, so predict_image() works on raw pixels
    os.makedirs("models", exist_ok=True)
    model.save("models/smartwaste_final.keras")
    print("Saved → models/smartwaste_final.keras")

    return model, class_names


# ── Inference helper ──────────────────────────────────────────────────────────

def predict_image(model: Model, class_names: list, img_path: str, top_k: int = 3):
    """
    Pass raw [0, 255] pixels directly to model.predict().
    Do NOT call preprocess_input manually — it lives inside the model graph.
    Calling it externally would apply it twice and shift pixel values outside
    the range EfficientNet expects, silently degrading predictions.
    """
    import matplotlib.pyplot as plt
    from tensorflow.keras.preprocessing import image

    img = image.load_img(img_path, target_size=(224, 224))
    arr = image.img_to_array(img)           # raw [0, 255]
    arr = np.expand_dims(arr, axis=0)       # (1, 224, 224, 3)

    # model graph: augmentation (no-op) → preprocess_input → base → head
    preds   = model.predict(arr, verbose=0)[0]
    top_idx = np.argsort(preds)[::-1][:top_k]

    plt.imshow(image.load_img(img_path))
    plt.axis("off")
    plt.title(os.path.basename(img_path))
    plt.tight_layout()
    plt.show()

    print(f"\nTop-{top_k} predictions:")
    for i in top_idx:
        print(f"  {class_names[i]:<12} {preds[i]*100:.2f}%")

    final_class = class_names[int(np.argmax(preds))]
    confidence  = float(np.max(preds)) * 100
    print(f"\nFinal: {final_class} ({confidence:.2f}%)")
    return final_class, confidence


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model, class_names = train()

    # Quick smoke-test on one image after training
    test_img = "data/split/test"
    first_class = os.listdir(test_img)[0]
    first_img   = os.listdir(os.path.join(test_img, first_class))[0]
    sample_path = os.path.join(test_img, first_class, first_img)
    predict_image(model, class_names, sample_path)