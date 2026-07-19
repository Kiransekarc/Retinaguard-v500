import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import cv2

# ================= CONFIGURATION =================
LATENT_DIM = 128
IMG_SIZE = (64, 64, 3)
BATCH_SIZE = 64
EPOCHS = 5000
N_CRITIC = 5
GP_WEIGHT = 10.0
# =================================================

def build_generator():
    """Builds the WGAN Generator model for synthetic fundus images."""
    model = models.Sequential(name="Generator")
    
    # Foundation
    model.add(layers.Dense(8 * 8 * 256, use_bias=False, input_shape=(LATENT_DIM,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((8, 8, 256)))
    
    # Upsampling block 1: 8x8 -> 16x16
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    # Upsampling block 2: 16x16 -> 32x32
    model.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    # Upsampling block 3: 32x32 -> 64x64
    model.add(layers.Conv2DTranspose(3, (4, 4), strides=(2, 2), padding='same', activation='tanh', use_bias=False))
    
    return model

def build_critic():
    """Builds the WGAN Critic model."""
    model = models.Sequential(name="Critic")
    
    # Downsampling block 1: 64x64 -> 32x32
    model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', input_shape=IMG_SIZE))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))
    
    # Downsampling block 2: 32x32 -> 16x16
    model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))
    
    # Downsampling block 3: 16x16 -> 8x8
    model.add(layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))
    
    # Output logic score
    model.add(layers.Flatten())
    model.add(layers.Dense(1)) # No sigmoid activation for WGAN critic
    
    return model

def wasserstein_loss(y_true, y_pred):
    """Calculates the Wasserstein loss."""
    return tf.reduce_mean(y_true * y_pred)

class WGAN_GP(tf.keras.Model):
    """Wasserstein GAN with Gradient Penalty for fundus image synthesis."""
    def __init__(self, critic, generator, latent_dim, critic_steps, gp_weight):
        super(WGAN_GP, self).__init__()
        self.critic = critic
        self.generator = generator
        self.latent_dim = latent_dim
        self.critic_steps = critic_steps
        self.gp_weight = gp_weight

    def compile(self, c_optimizer, g_optimizer):
        super(WGAN_GP, self).compile()
        self.c_optimizer = c_optimizer
        self.g_optimizer = g_optimizer
        self.c_loss_metric = tf.keras.metrics.Mean(name="c_loss")
        self.g_loss_metric = tf.keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.c_loss_metric, self.g_loss_metric]

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """Calculates the gradient penalty."""
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.critic(interpolated, training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    @tf.function
    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]
            
        batch_size = tf.shape(real_images)[0]

        # 1. Train Critic
        for i in range(self.critic_steps):
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            with tf.GradientTape() as tape:
                fake_images = self.generator(random_latent_vectors, training=True)
                fake_logits = self.critic(fake_images, training=True)
                real_logits = self.critic(real_images, training=True)

                # Wasserstein loss
                c_cost = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)
                
                # Gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                
                # Total loss
                c_loss = c_cost + self.gp_weight * gp

            c_gradient = tape.gradient(c_loss, self.critic.trainable_variables)
            self.c_optimizer.apply_gradients(zip(c_gradient, self.critic.trainable_variables))

        # 2. Train Generator
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_latent_vectors, training=True)
            fake_logits = self.critic(fake_images, training=True)
            
            # Generator wants critic to predict real (negative logits)
            g_loss = -tf.reduce_mean(fake_logits)
            
        g_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_gradient, self.generator.trainable_variables))

        self.c_loss_metric.update_state(c_loss)
        self.g_loss_metric.update_state(g_loss)
        
        return {
            "c_loss": self.c_loss_metric.result(),
            "g_loss": self.g_loss_metric.result()
        }

if __name__ == "__main__":
    print("Initializing WGAN-GP Architecture for Synthetic Fundus Generation...")
    gen = build_generator()
    crit = build_critic()
    
    wgan = WGAN_GP(
        critic=crit,
        generator=gen,
        latent_dim=LATENT_DIM,
        critic_steps=N_CRITIC,
        gp_weight=GP_WEIGHT
    )
    
    # Use Adam optimizer as recommended by WGAN-GP paper
    wgan.compile(
        c_optimizer=Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9),
        g_optimizer=Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
    )
    print("Architecture Ready.")
