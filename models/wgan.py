import json

code = """# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from google.colab import drive
import os
import time

# ========================
# GPU CHECK AND SETUP
# ========================
def check_gpu_setup():
    print("=== GPU SETUP CHECK ===")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU memory growth enabled")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("⚠️ Running on CPU (Slow) - PLEASE ENABLE GPU IN RUNTIME SETTINGS")
    print("=" * 50)

check_gpu_setup()

# ========================
# 1. MOUNT DRIVE (CRITICAL STEP)
# ========================
print("📂 Mounting Google Drive...")
try:
    drive.mount('/content/drive', force_remount=True)
    print("✅ Drive Mounted Successfully!")
except Exception as e:
    print(f"⚠️ Drive Mount Error: {e}")

# PATHS
train_dataset_path = "/content/drive/MyDrive/Dataset/Train/Retinitis Pigmentosa"
save_dir_base = "/content/drive/MyDrive/WGAN_Results"
os.makedirs(save_dir_base, exist_ok=True)

# ========================
# 2. PARAMETERS
# ========================
IMG_SHAPE = (128, 128, 3)
LATENT_DIM = 128
BATCH_SIZE = 32
EPOCHS = 5000
SAVE_INTERVAL = 10
DISPLAY_INTERVAL = 10

# WGAN Hyperparameters
GP_WEIGHT = 10.0
CRITIC_STEPS = 5
INITIAL_LR = 0.0001

print(f"Configuration: {EPOCHS} Epochs | Batch Size {BATCH_SIZE}")

# ========================
# 3. BUILD MODELS
# ========================
def build_generator():
    model = Sequential()
    model.add(Input(shape=(LATENT_DIM,)))
    model.add(Dense(128 * 16 * 16))
    model.add(Reshape((16, 16, 128)))
    model.add(BatchNormalization(momentum=0.8))

    # Upsampling blocks
    model.add(Conv2DTranspose(128, 4, strides=2, padding="same", activation='relu'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2DTranspose(64, 4, strides=2, padding="same", activation='relu'))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2DTranspose(32, 4, strides=2, padding="same", activation='relu'))
    model.add(BatchNormalization(momentum=0.8))

    # Output
    model.add(Conv2DTranspose(3, 4, strides=1, padding="same", activation='tanh'))
    return model

def build_critic():
    model = Sequential()
    model.add(Input(shape=IMG_SHAPE))
    model.add(Conv2D(64, 4, strides=2, padding="same"))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(128, 4, strides=2, padding="same"))
    model.add(LayerNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(256, 4, strides=2, padding="same"))
    model.add(LayerNormalization())
    model.add(LeakyReLU(0.2))

    model.add(Flatten())
    model.add(Dense(1)) # Linear output
    return model

generator = build_generator()
critic = build_critic()

# Optimizers
opt_g = Adam(learning_rate=INITIAL_LR, beta_1=0.5, beta_2=0.9)
opt_d = Adam(learning_rate=INITIAL_LR, beta_1=0.5, beta_2=0.9)

# ========================
# 4. TRAINING STEPS
# ========================
@tf.function
def gradient_penalty(critic, real, fake):
    batch_size = tf.shape(real)[0]
    epsilon = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
    interpolated = epsilon * real + (1 - epsilon) * fake

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = critic(interpolated, training=True)

    grads = gp_tape.gradient(pred, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp

@tf.function
def train_step(real_images):
    batch_size = tf.shape(real_images)[0]

    # Train Critic
    for _ in range(CRITIC_STEPS):
        noise = tf.random.normal([batch_size, LATENT_DIM])
        with tf.GradientTape() as tape:
            fake_images = generator(noise, training=True)
            real_logits = critic(real_images, training=True)
            fake_logits = critic(fake_images, training=True)

            d_cost = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)
            gp = gradient_penalty(critic, real_images, fake_images)
            d_loss = d_cost + GP_WEIGHT * gp

        d_grads = tape.gradient(d_loss, critic.trainable_variables)
        opt_d.apply_gradients(zip(d_grads, critic.trainable_variables))

    # Train Generator
    noise = tf.random.normal([batch_size, LATENT_DIM])
    with tf.GradientTape() as tape:
        fake_images = generator(noise, training=True)
        gen_logits = critic(fake_images, training=True)
        g_loss = -tf.reduce_mean(gen_logits)

    g_grads = tape.gradient(g_loss, generator.trainable_variables)
    opt_g.apply_gradients(zip(g_grads, generator.trainable_variables))

    return d_loss, g_loss, tf.reduce_mean(real_logits), tf.reduce_mean(fake_logits)

# ========================
# 5. AUTO-CORRECTION & HELPERS
# ========================
class ModelBackup:
    def __init__(self):
        self.gen_weights = None
        self.critic_weights = None
    def backup(self, gen, cri):
        self.gen_weights = gen.get_weights()
        self.critic_weights = cri.get_weights()
    def restore(self, gen, cri):
        print("   ♻️ Restoring weights from last good epoch...")
        gen.set_weights(self.gen_weights)
        cri.set_weights(self.critic_weights)

def save_images(epoch, num=5):
    noise = np.random.normal(0, 1, (num, LATENT_DIM))
    imgs = generator.predict(noise, verbose=0) * 0.5 + 0.5
    fig, axs = plt.subplots(1, num, figsize=(15, 3))
    if num==1: axs=[axs]
    for i in range(num):
        axs[i].imshow(imgs[i])
        axs[i].axis('off')
    plt.savefig(f"{save_dir_base}/epoch_{epoch}.png", bbox_inches='tight')
    plt.close()
    print(f"   💾 Images saved for Epoch {epoch}")

def save_models(epoch):
    generator.save(f"{save_dir_base}/gen_{epoch}.h5")
    critic.save(f"{save_dir_base}/crit_{epoch}.h5")

def display_training_images(epoch, num=5):
    noise = np.random.normal(0, 1, (num, LATENT_DIM))
    imgs = generator.predict(noise, verbose=0) * 0.5 + 0.5
    fig, axs = plt.subplots(1, num, figsize=(15, 3))
    if num==1: axs=[axs]
    for i in range(num):
        axs[i].imshow(imgs[i])
        axs[i].axis('off')
    plt.show()

def load_data():
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    imgs = []
    print(f"Loading from {train_dataset_path}...")
    if not os.path.exists(train_dataset_path):
        print(f"❌ ERROR: Folder not found: {train_dataset_path}")
        return np.array([])

    for root, _, files in os.walk(train_dataset_path):
        for f in files:
            if f.lower().endswith(('.jpg','.png','.jpeg')):
                try:
                    p = os.path.join(root, f)
                    i = load_img(p, target_size=IMG_SHAPE[:2])
                    i = img_to_array(i)/127.5 - 1.0
                    imgs.append(i)
                except: pass
    return np.array(imgs)

# ========================
# 6. MAIN TRAINING LOOP
# ========================
def train_robust(dataset):
    global generator, critic

    start_epoch = 0
    print(f"🔍 Checking for existing checkpoints in: {save_dir_base}")

    existing_files = os.listdir(save_dir_base)
    checkpoints = [f for f in existing_files if f.startswith('gen_') and f.endswith('.h5')]

    if len(checkpoints) > 0:
        valid_epochs = []
        for f in checkpoints:
            try:
                num_part = f.split('_')[1].split('.')[0]
                if num_part.isdigit():
                    valid_epochs.append(int(num_part))
            except: pass

        if len(valid_epochs) > 0:
            latest_epoch = max(valid_epochs)
            print(f"   ✅ Found checkpoint from Epoch {latest_epoch}!")
            try:
                generator.load_weights(f"{save_dir_base}/gen_{latest_epoch}.h5")
                critic.load_weights(f"{save_dir_base}/crit_{latest_epoch}.h5")
                start_epoch = latest_epoch
                print(f"   🚀 Resuming training from Epoch {start_epoch + 1}")
            except Exception as e:
                print("   ⚠️ Could not load checkpoint.")
        else:
             print("   ℹ️ No valid numbered checkpoints found.")
    else:
        print("   🆕 No checkpoints found. Starting from scratch.")

    backup_system = ModelBackup()
    backup_system.backup(generator, critic)
    batch_count = dataset.shape[0] // BATCH_SIZE

    for epoch in range(start_epoch, EPOCHS):
        start_time = time.time()
        np.random.shuffle(dataset)
        backup_system.backup(generator, critic)

        epoch_d = 0; epoch_g = 0; real_sc = 0; fake_sc = 0
        crash_detected = False

        for i in range(batch_count):
            real_imgs = dataset[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
            real_imgs = tf.convert_to_tensor(real_imgs, dtype=tf.float32)

            try:
                d_loss, g_loss, r_score, f_score = train_step(real_imgs)
                if tf.math.is_nan(d_loss) or tf.math.is_nan(g_loss):
                    raise ValueError("Loss Exploded")
                epoch_d += float(d_loss); epoch_g += float(g_loss); real_sc += float(r_score); fake_sc += float(f_score)
            except Exception as e:
                crash_detected = True
                break

        if crash_detected:
            backup_system.restore(generator, critic)
            current_lr = float(opt_g.learning_rate.numpy())
            new_lr = current_lr * 0.5
            if new_lr < 1e-6:
                generator = build_generator(); critic = build_critic(); new_lr = INITIAL_LR
            opt_g.learning_rate.assign(new_lr); opt_d.learning_rate.assign(new_lr)
            continue

        if (epoch + 1) % SAVE_INTERVAL == 0:
            save_images(epoch + 1)
            save_models(epoch + 1)

if __name__ == "__main__":
    try:
        data = load_data()
        if len(data) == 0:
            data = np.random.normal(0, 1, (100, 128, 128, 3))
        train_robust(data)
    except KeyboardInterrupt:
        generator.save(f"{save_dir_base}/gen_interrupted.h5")
"""

blocks = code.split("# ========================")

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

for i, block in enumerate(blocks):
    if not block.strip():
        continue
    notebook["cells"].append({
        "cell_type": "code",
        "metadata": {},
        "execution_count": i,
        "outputs": [],
        "source": [line + "\\n" for line in block.strip().split("\\n")]
    })

with open("Original_WGAN_Training.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=4)
print("Saved!")
