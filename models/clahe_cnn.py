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
import cv2  # <--- Needed for the fix
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

# ================= FYP CONFIGURATION =================
REAL_RP_PATH = "/content/drive/MyDrive/Dataset/Train/Retinitis Pigmentosa"
FAKE_RP_PATH = "/content/drive/MyDrive/Final_RP_Dataset"
NORMAL_PATH  = "/content/drive/MyDrive/Dataset/Train/Normal"

PROJECT_ROOT = "/content/drive/MyDrive/RP_Classification_Experiment"
MODEL_SAVE_PATH = os.path.join(PROJECT_ROOT, "Models", "RP_Classifier_FYP.h5")
LOG_FILE        = os.path.join(PROJECT_ROOT, "training_log_enhanced.csv")
ACCURACY_PLOT   = os.path.join(PROJECT_ROOT, "Accuracy_Graph_Enhanced.png")
CONFUSION_PLOT  = os.path.join(PROJECT_ROOT, "Confusion_Matrix_Enhanced.png")

IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 20

# ================= THE FIX: CUSTOM PREPROCESSING =================
def apply_clahe_training(img):
    \"\"\"
    Applies CLAHE to training images on the fly.
    This forces the model to learn that 'Sharp Veins' are NOT diseases.
    \"\"\"
    # Convert RGB to LAB (img comes in as RGB from Keras)
    # Note: Keras loads images as RGB (0-255) but as floats or ints depending.
    # We force uint8 for OpenCV processing
    img = img.astype('uint8')

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

    # Return as float 0-1 for the network (Standard Keras format)
    return final.astype('float32') / 255.0

# =================================================================

def train_classifier():
    print("🚀 Starting RETRAINING with CLAHE Enhancement...")

    if not os.path.exists(os.path.join(PROJECT_ROOT, "Models")):
        os.makedirs(os.path.join(PROJECT_ROOT, "Models"))

    # A. Load Filenames
    try:
        real_rp = [os.path.join(REAL_RP_PATH, f) for f in os.listdir(REAL_RP_PATH) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        fake_rp = [os.path.join(FAKE_RP_PATH, f) for f in os.listdir(FAKE_RP_PATH) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        normal  = [os.path.join(NORMAL_PATH, f) for f in os.listdir(NORMAL_PATH) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    except Exception as e:
        print(f"❌ Error reading files: {e}")
        return

    # B. The Split
    train_rp, test_rp = train_test_split(real_rp, test_size=0.2, random_state=42)
    train_norm, test_norm = train_test_split(normal, test_size=0.2, random_state=42)
    full_train_rp = train_rp + fake_rp

    print(f"📊 Training on: {len(full_train_rp)} RP vs {len(train_norm)} Normal")

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

    # D. Generators with THE FIX
    # We REMOVE 'rescale=1./255' because our custom function handles the normalization
    train_datagen = ImageDataGenerator(
        preprocessing_function=apply_clahe_training, # <--- The Magic Line
        rotation_range=20,
        horizontal_flip=True,
        zoom_range=0.1
    )

    test_datagen  = ImageDataGenerator(
        preprocessing_function=apply_clahe_training # <--- Apply to test data too!
    )

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

    # E. Model
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

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])

    # F. Train
    print("\\n🏋️ Training Started (This might take a minute longer due to enhancement)...")
    history = model.fit(train_gen, epochs=EPOCHS, validation_data=test_gen, callbacks=[CSVLogger(LOG_FILE)])

    # G. Save
    model.save(MODEL_SAVE_PATH)
    print(f"\\n✅ Enhanced Model Saved: {MODEL_SAVE_PATH}")

    # Report
    print("🔍 Generating Confusion Matrix...")
    predictions = model.predict(test_gen)
    y_pred = (predictions > 0.5).astype(int).flatten()
    y_true = test_gen.classes
    class_labels = list(test_gen.class_indices.keys())

    print("\\n" + "="*40)
    print("🏆 FINAL REPORT (Enhanced Model)")
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
                "# RetinaGuard: Enhanced CNN Experiment (Phase 2)\\n",
                "This notebook documents the second phase of the project where we attempted to fix the flaws of the basic CNN by applying **CLAHE (Contrast Limited Adaptive Histogram Equalization)** to the images.\\n",
                "\\n",
                "**Conclusion drawn from this experiment:** While CLAHE helped the CNN distinguish sharp veins from disease markers, the 64x64 resolution and shallow architecture were still limiting factors. This ultimately led to the final decision to use Transfer Learning (ResNet50V2 at 224x224)."
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

with open("Enhanced_CNN_CLAHE_Experiment.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=4)
print("Saved CLAHE Notebook!")
