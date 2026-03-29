# SmartWaste/src/pipeline.py

import mlflow
from src.data_preprocessing import split_dataset, preprocess
from src.model_training import train
from src.model_evaluation import evaluate

if __name__ == "__main__":
    with mlflow.start_run(run_name="full_pipeline"):

        print("Splitting raw images into train/val/test...")
        split_dataset("data/raw", "data/split")

        print("Running preprocessing (building tf.data pipelines)...")
        train_ds, val_ds, test_ds, class_names = preprocess("data/split")

        print("Running training (Stage 1 + Stage 2 fine-tuning)...")
        model, class_names = train()

        print("Running evaluation on held-out test set...")
        evaluate()