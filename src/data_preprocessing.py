# SmartWaste/src/data_preprocessing.py

import os 
import shutil
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

# ── Constants ─────────────────────────────────────────────────────────────────

IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
SEED       = 42
AUTOTUNE   = tf.data.AUTOTUNE


# ── Dataset split ─────────────────────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  
def split_dataset(
    raw_dir: str = "data/raw",
    out_dir: str = "data/split",
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> None:
    """
    Copies data/raw/<class>/*.jpg into data/split/{train,val,test}/<class>/
    Idempotent — skips if the split already exists.
    Call once before anything else.
    """
    out_path = Path(out_dir)
    if (out_path / "train").exists():
        print("Split already exists — skipping.")
        return

    valid_ext = {".jpg", ".jpeg", ".png", ".webp"}
    raw_path  = Path(raw_dir)

    for cls_dir in sorted(raw_path.iterdir()):
        if not cls_dir.is_dir():
            continue

        images = [p for p in cls_dir.glob("*.*") if p.suffix.lower() in valid_ext]
        if not images:
            print(f"  Warning: no images found in {cls_dir}")
            continue

        train_imgs, temp = train_test_split(
            images, test_size=val_ratio + test_ratio, random_state=SEED
        )
        val_imgs, test_imgs = train_test_split(
            temp,
            test_size=test_ratio / (val_ratio + test_ratio),
            random_state=SEED,
        )

        for split_name, split_imgs in [
            ("train", train_imgs),
            ("val",   val_imgs),
            ("test",  test_imgs),
        ]:
            dest = out_path / split_name / cls_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            for img_path in split_imgs:
                shutil.copy(img_path, dest / img_path.name)

    print(f"Split complete → {out_dir}/{{train, val, test}}")


# ── tf.data pipeline ──────────────────────────────────────────────────────────

def _make_dataset(directory: str, shuffle: bool) -> tf.data.Dataset:
    """Loads a directory into a tf.data.Dataset. No augmentation, no scaling."""
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED,
        label_mode="int",           # sparse labels — compatible with sparse_categorical_crossentropy
    )


def _augmentation_layer() -> tf.keras.Sequential:
    """
    Augmentation lives INSIDE the model graph (applied only during training).
    Defined here so preprocessing and model stay in sync — train.py imports
    this directly and inserts it as the first layer.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.2),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
        tf.keras.layers.RandomBrightness(0.2),      # handles outdoor/indoor lighting variation
    ], name="augmentation")


def _count_images(directory: str) -> int:
    """Counts all image files across class subdirectories."""
    valid_ext = {".jpg", ".jpeg", ".png", ".webp"}
    return sum(
        1
        for cls_dir in Path(directory).iterdir() if cls_dir.is_dir()
        for f in cls_dir.iterdir() if f.suffix.lower() in valid_ext
    )


def preprocess(split_dir: str = "data/split"):
    train_dir = os.path.join(split_dir, "train")
    val_dir   = os.path.join(split_dir, "val")
    test_dir  = os.path.join(split_dir, "test")

    for d in [train_dir, val_dir, test_dir]:
        if not os.path.exists(d):
            raise FileNotFoundError(
                f"Directory not found: {d}\n"
                "Run split_dataset() first, or check your split_dir path."
            )

    train_ds = _make_dataset(train_dir, shuffle=True)
    val_ds   = _make_dataset(val_dir,   shuffle=False)
    test_ds  = _make_dataset(test_dir,  shuffle=False)

    class_names = train_ds.class_names
    print("Classes :", class_names)

    # FIX: count from disk — cardinality() counts batches, not images
    print(f"Samples → train: {_count_images(train_dir)}, "
          f"val: {_count_images(val_dir)}, "
          f"test: {_count_images(test_dir)}")

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds   = val_ds.prefetch(AUTOTUNE)
    test_ds  = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names

# ── Class weights ─────────────────────────────────────────────────────────────

def compute_class_weights(train_ds: tf.data.Dataset) -> dict:
    """
    Extracts labels from train_ds and computes balanced class weights.
    Works correctly whether train_ds is prefetched or not.
    """
    # unbatch() yields individual (image, label) pairs where label is a scalar tensor
    # np.concatenate fails on 0-d arrays — use np.array() to collect scalars instead
    y_labels = np.array([y.numpy() for _, y in train_ds.unbatch()])

    classes     = np.unique(y_labels)
    weights     = compute_class_weight("balanced", classes=classes, y=y_labels)
    weight_dict = dict(enumerate(weights))
    print("Class weights:", {int(k): round(float(v), 4) for k, v in weight_dict.items()})
    return weight_dict
# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    split_dataset("data/raw", "data/split")
    train_ds, val_ds, test_ds, class_names = preprocess("data/split")
    compute_class_weights(train_ds)