import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Conv2DTranspose, Conv2D, Flatten, LeakyReLU, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from keras.models import load_model
from data_loader import load_data_npz
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import sys
from datetime import datetime

# List all available GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        print(f"TensorFlow is using GPU: {gpu}")
else:
    print("No GPU found. TensorFlow is using the CPU.")


current_dir = os.path.dirname(os.path.abspath(__file__))
output_images_dir = os.path.join(current_dir, 'output_images')

models_dir = os.path.join(current_dir, 'models')

loss_dir = os.path.join(current_dir, 'loss')

parent_dir = os.path.dirname(current_dir)
grandpa_dir = os.path.dirname(parent_dir)
DNS_IP_directory = os.path.join(grandpa_dir, 'DNS_IP')
GERACAO_DE_PACOTES_PELA_GAN_directory = os.path.join(DNS_IP_directory, 'GERAÇÃO_DE_PACOTES_PELA_GAN')

weights_dir = os.path.join(current_dir, 'weights')
generator_weights_dir = os.path.join(weights_dir, 'generator')
discriminator_weights_dir = os.path.join(weights_dir, 'discriminator')

if not os.path.exists(f"{generator_weights_dir}"):
    os.makedirs(f"{generator_weights_dir}")
        
if not os.path.exists(f"{discriminator_weights_dir}"):
    os.makedirs(f"{discriminator_weights_dir}")


# Definir o diretório de logs
log_dir = os.path.join(current_dir, "logs", datetime.now().strftime("%Y%m%d-%H%M%S"))

if not os.path.exists(f"{log_dir}"):
    os.makedirs(f"{log_dir}")

# Criar o callback do TensorBoard
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True,
    update_freq='epoch')

FINE_TUNING_dir = os.path.join(grandpa_dir, 'FINE-TUNING')
NOVO_FINE_TUNING_dir = os.path.join(FINE_TUNING_dir, 'NOVO FINE-TUNING')
pre_trained_models_weights_dir = os.path.join(NOVO_FINE_TUNING_dir, 'weights')

import keras

if not os.path.exists(f"{models_dir}"):
        os.makedirs(f"{models_dir}")
        
if not os.path.exists(f"{loss_dir}"):
        os.makedirs(f"{loss_dir}")
        
        
l2_reg = 2.5e-5
activation = LeakyReLU(alpha=0.2)
learning_rate = 0.001
beta1 = 0.5

(x_train, _), (_, _) = load_data_npz()
x_train = np.expand_dims(x_train, axis=-1)
x_train = (x_train / 255.0) * 2.0 - 1.0


class WGAN(keras.Model):
    def __init__(
        self,
        discriminator,
        generator,
        latent_dim,
        discriminator_extra_steps=5,
        gp_weight=10.0,
    ):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight
        self.current_g_loss = 0.0
        self.current_d_loss = 0.0

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    @tf.function
    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]

        # Get the batch size
        batch_size = tf.shape(real_images)[0]

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator(random_latent_vectors, training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator(fake_images, training=True)
                # Get the logits for the real images
                real_logits = self.discriminator(real_images, training=True)

                # Calculate the discriminator loss using the fake and real image logits
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )
            

        # Train the generator
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(random_latent_vectors, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(generated_images, training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        
        self.current_g_loss = g_loss
        self.current_d_loss = d_loss    
            
        return {"d_loss": d_loss, "g_loss": g_loss}
    
    
def save_generated_images(epoch, generator, batch, examples=10, random_dim=1024):
    """Saves generated images to a file.
    
    Args:
        epoch (integer): Epoch number
        generator (Sequential): Generator model
        batch (integer): Batch number
        examples (integer): Number of examples
        random_dim (integer): Dimension of the latent vector
    """
    sys.path.append(GERACAO_DE_PACOTES_PELA_GAN_directory)
    from packets_generation import save_packets_on_training
    
    noise = np.random.normal(0, 1, (examples, random_dim))

    generated_images = generator.predict(noise)

    generated_images = (generated_images + 1) * 127.5
    generated_images = generated_images.astype(np.uint8)
    
    if not os.path.exists(f"{output_images_dir}/generated_images_{epoch}"):
        os.makedirs(f"{output_images_dir}/generated_images_{epoch}")

    plt.figure(figsize=(20, 4))
    for i in range(examples):
        image = generated_images[i, :, :, :]
        image = np.reshape(image, [28, 28])
        plt.subplot(1, examples, i+1)
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        
    plt.savefig(f"{output_images_dir}/generated_images_{epoch}/generated_image_{batch}.png")
    save_packets_on_training(generated_images, f"generated_images_{batch}", epoch, examples)
    plt.close()
    
    
# Atualize a classe GANMonitor
class GANMonitor(tf.keras.callbacks.Callback):
    def __init__(self, log_dir):
        super(GANMonitor, self).__init__()
        self.log_dir = log_dir
        self.file_writer = tf.summary.create_file_writer(log_dir)
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        d_loss = logs.get('d_loss', self.model.current_d_loss)
        g_loss = logs.get('g_loss', self.model.current_g_loss)
        
        
        with self.file_writer.as_default():
            tf.summary.scalar('g_loss', g_loss, step=epoch)
            tf.summary.scalar('d_loss', d_loss, step=epoch)
        self.file_writer.flush()
        
        save_generated_images(epoch, self.model.generator, epoch, 10)
        self.model.generator.save(f"{models_dir}/generator_model{epoch}.keras")
        self.model.generator.save_weights(f"{generator_weights_dir}/generator_weights{epoch}.weights.h5")
        self.model.discriminator.save_weights(f"{discriminator_weights_dir}/discriminator_weights{epoch}.weights.h5")
        
        
def build_generator(random_dim):
    """Builds the generator model.
    
    Args:
        random_dim (integer): Dimension of the latent vector
        
    Returns:
        Sequential: Generator model
    """
    model = tf.keras.Sequential()

    model.add(Dense(64, input_dim=random_dim, activation=activation, kernel_regularizer=l2(l2_reg)))
    model.add(Dense(1024, activation=activation, kernel_regularizer=l2(l2_reg)))
    
    model.add(Dense(12544, activation=activation, kernel_regularizer=l2(l2_reg))),

    model.add(Reshape((7, 7, 256)))

    model.add(Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', activation=activation, kernel_regularizer=l2(l2_reg)))

    model.add(Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same', activation=activation, kernel_regularizer=l2(l2_reg)))

    model.add(Conv2D(1, (1, 1), padding='same', activation='tanh'))

    return model
            
            
def build_discriminator(input_shape):
    """Builds the discriminator model.
    
    Args:
        input_shape (tuple): Shape of the input images
        
    Returns:
        Sequential: Discriminator model
    """
    model = tf.keras.Sequential()

    model.add(Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding='same', input_shape=input_shape, kernel_regularizer=l2(l2_reg), name='conv1'))
    model.add(tf.keras.layers.Activation('relu'))
    
    model.add(Conv2D(128, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_regularizer=l2(l2_reg), name='conv2'))
    model.add(tf.keras.layers.Activation('relu'))

    model.add(Flatten(name='flatten'))

    model.add(Dense(1, activation='linear', kernel_regularizer=l2(l2_reg), name='dense'))

    return model


    
generator_optimizer = keras.optimizers.Adam(
    learning_rate=learning_rate, beta_1=beta1
)
discriminator_optimizer = keras.optimizers.Adam(
    learning_rate=learning_rate, beta_1=beta1
)


# Define the loss functions for the discriminator,
# which should be (fake_loss - real_loss).
def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)


epochs = 50

input_shape = (28, 28, 1)
random_dim = 1024

batch_size = 1024

tensorboard_callback = GANMonitor(log_dir)

wgan = WGAN(
    discriminator=build_discriminator(input_shape),
    generator=build_generator(random_dim),
    latent_dim=random_dim,
    discriminator_extra_steps=5,
)

wgan.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
    g_loss_fn=generator_loss,
    d_loss_fn=discriminator_loss,
)

history = wgan.fit(x_train, batch_size=batch_size, epochs=epochs, callbacks=[tensorboard_callback])
