import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import os
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
# ================= CONFIGURATION =================
# 1. Path to your BEST model (Epoch 4260)
MODEL_PATH = "/content/drive/MyDrive/WGAN_Results/gen_4260.h5"

# 2. Where to save the final images
OUTPUT_FOLDER = "/content/drive/MyDrive/Final_RP_Dataset"

# 3. Settings
NUM_IMAGES = 1000
LATENT_DIM = 128    # Confirmed 128
IMG_SIZE = (64, 64) # Final size
# =================================================

def generate_dataset():
    # 1. Create Folder
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"📁 Created output folder: {OUTPUT_FOLDER}")
    else:
        print(f"📁 Saving to existing folder: {OUTPUT_FOLDER}")

    # 2. Load Model
    print(f"⏳ Loading model from: {MODEL_PATH}...")
    try:
        generator = load_model(MODEL_PATH, compile=False)
        print("✅ Model loaded!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    print(f"🚀 Starting generation of {NUM_IMAGES} images...")

    # 3. Generate in Batches (Safer for RAM)
    batch_size = 100
    total_batches = NUM_IMAGES // batch_size

    count = 0
    for b in range(total_batches):
        # Generate Noise
        noise = np.random.normal(0, 1, (batch_size, LATENT_DIM))

        # Predict
        gen_imgs = generator.predict(noise, verbose=0)

        # Rescale to 0-255
        gen_imgs = 127.5 * gen_imgs + 127.5
        gen_imgs = gen_imgs.astype(np.uint8)

        # Process and Save
        for i in range(batch_size):
            img = gen_imgs[i]

            # RESIZE (Ensure it is 64x64)
            img = cv2.resize(img, IMG_SIZE)

            # COLOR FIX (RGB to BGR for OpenCV saving)
            # Since we proved model is RGB, we MUST convert to BGR before saving with cv2
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # SMOOTHING (Remove Grid Pattern) 🧹
            img = cv2.GaussianBlur(img, (3, 3), 0)

            # Save
            filename = f"{OUTPUT_FOLDER}/RP_Gen_{count+1:04d}.png"
            cv2.imwrite(filename, img)
            count += 1

        print(f"   Saved batch {b+1}/{total_batches} ({count} images total)")

    print(f"\n🎉 DONE! You now have {count} images in '{OUTPUT_FOLDER}'")

if __name__ == "__main__":
    generate_dataset()
