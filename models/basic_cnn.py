import json

code = """import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import CSVLogger
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
# ================= FYP CONFIGURATION =================
# 1. INPUT PATHS
REAL_RP_PATH = "/content/drive/MyDrive/Dataset/Train/Retinitis Pigmentosa"
FAKE_RP_PATH = "/content/drive/MyDrive/Final_RP_Dataset"
NORMAL_PATH  = "/content/drive/MyDrive/Dataset/Train/Normal"

# 2. OUTPUT PATHS
PROJECT_ROOT = "/content/drive/MyDrive/RP_Classification_Experiment"
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "Models", "RP_Classifier_FYP.h5")
LOG_FILE        = os.path.join(PROJECT_ROOT, "training_log.csv") # <--- EXCEL LOG FILE
ACCURACY_PLOT   = os.path.join(PROJECT_ROOT, "Accuracy_Graph.png")
CONFUSION_PLOT  = os.path.join(PROJECT_ROOT, "Confusion_Matrix.png")

# 3. Settings
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 20
# ======================================================

def setup_timeline_folders():
    if not os.path.exists(os.path.join(PROJECT_ROOT, "Models")):
        os.makedirs(os.path.join(PROJECT_ROOT, "Models"))
        print(f"✅ Created Timeline Folder: {PROJECT_ROOT}")

def train_classifier():
    print("🚀 Starting FYP Classification (With Detailed Logs)...")
    setup_timeline_folders()

    # --- SAFETY CHECK ---
    if not os.path.exists(NORMAL_PATH):
        print(f"\\n❌ STOP: I cannot find 'Normal' images at: {NORMAL_PATH}")
        return

    # A. Load Filenames
    try:
        real_rp = [os.path.join(REAL_RP_PATH, f) for f in os.listdir(REAL_RP_PATH) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        fake_rp = [os.path.join(FAKE_RP_PATH, f) for f in os.listdir(FAKE_RP_PATH) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        normal  = [os.path.join(NORMAL_PATH, f) for f in os.listdir(NORMAL_PATH) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    except Exception as e:
        print(f"❌ Error reading files: {e}")
        return

    # B. The Split (80/20)
    train_rp, test_rp = train_test_split(real_rp, test_size=0.2, random_state=42)
    train_norm, test_norm = train_test_split(normal, test_size=0.2, random_state=42)

    full_train_rp = train_rp + fake_rp

    print(f"📊 DATASET STATISTICS:")
    print(f"   TRAINING: {len(full_train_rp)} RP vs {len(train_norm)} Normal")
    print(f"   TESTING:  {len(test_rp)} RP vs {len(test_norm)} Normal")

    # C. Dataframes
    train_df = pd.DataFrame({
        'filename': full_train_rp + train_norm,
        'label': ['RP'] * len(full_train_rp) + ['Normal'] * len(train_norm)
    })

    test_df = pd.DataFrame({
        'filename': test_rp + test_norm,
        'label': ['RP'] * len(test_rp) + ['Normal'] * len(test_norm)
    })

    train_df = train_df.sample(frac=1).reset_index(drop=True)

    # D. Generators
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, horizontal_flip=True, zoom_range=0.1)
    test_datagen  = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_dataframe(
        train_df, x_col='filename', y_col='label',
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='binary'
    )

    test_gen = test_datagen.flow_from_dataframe(
        test_df, x_col='filename', y_col='label',
        target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='binary', shuffle=False
    )

    # E. Model Architecture
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    # ---------------------------------------------------------
    # 🔧 UPGRADE: We add Precision and Recall to the metrics
    # ---------------------------------------------------------
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])

    # F. Logging Callback (Saves Excel-ready data)
    csv_logger = CSVLogger(LOG_FILE, append=False)

    # G. Train
    print("\\n🏋️ Training Started (Watch the Precision/Recall columns!)...")
    history = model.fit(train_gen,
                        epochs=EPOCHS,
                        validation_data=test_gen,
                        callbacks=[csv_logger]) # <--- Added Logger

    # H. Save Model
    model.save(MODEL_SAVE_PATH)
    print(f"\\n✅ Model Saved: {MODEL_SAVE_PATH}")
    print(f"📝 detailed Log Saved: {LOG_FILE}")

    # ================= REPORTING =================

    # 1. Accuracy Graph
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Test Accuracy')
    plt.title('Model Accuracy per Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(ACCURACY_PLOT)
    print(f"📈 Graph Saved: {ACCURACY_PLOT}")

    # 2. Confusion Matrix
    print("🔍 Generating Confusion Matrix...")
    predictions = model.predict(test_gen)
    y_pred = (predictions > 0.5).astype(int).flatten()
    y_true = test_gen.classes

    class_labels = list(test_gen.class_indices.keys())

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(CONFUSION_PLOT)
    print(f"🟦 Matrix Saved: {CONFUSION_PLOT}")

    print("\\n" + "="*40)
    print("🏆 FINAL YEAR PROJECT EVALUATION REPORT")
    print("="*40)
    print(classification_report(y_true, y_pred, target_names=class_labels))
    print("="*40)

if __name__ == "__main__":
    train_classifier()
"""

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# RetinaGuard: Basic CNN Experiment (Phase 1)\\n",
                "This notebook documents the initial phase of the project where a standard 64x64 Convolutional Neural Network was tested for RP detection. \\n",
                "\\n",
                "**Conclusion drawn from this experiment:** The basic CNN struggled to capture the highly subtle micro-patterns of Retinitis Pigmentosa due to the downscaled 64x64 resolution and shallow architecture. This proved the necessity of upgrading to our final Transfer Learning architecture (ResNet50V2 at 224x224) which is documented in the final classifier notebook."
            ]
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": 1,
            "outputs": [],
            "source": [line + "\\n" for line in code.split("\\n")]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open("Basic_CNN_Experiment.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=4)
print("Saved CNN Notebook!")
