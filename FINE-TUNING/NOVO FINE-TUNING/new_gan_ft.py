import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Conv2DTranspose, Conv2D, Flatten, LeakyReLU, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from torch import dropout
from data_loader import load_data_npz
import numpy as np
import matplotlib.pyplot as plt
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
output_images_dir = os.path.join(current_dir, 'output_images')

models_dir = os.path.join(current_dir, 'models')

loss_dir = os.path.join(current_dir, 'loss')

if not os.path.exists(f"{models_dir}"):
        os.makedirs(f"{models_dir}")
        
if not os.path.exists(f"{loss_dir}"):
        os.makedirs(f"{loss_dir}")

def compute_gradient_penalty(real_images, fake_images):
    """Calculates the gradient penalty for a batch of "real" and "fake" images, as a loss function for the discriminator.
    
    Args:
        real_images (tensor): Real images
        fake_images (tensor): Fake images
        
    Returns:
        tensor: Gradient penalty
    """
    alpha = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0.0, maxval=1.0)
    interpolated_samples = alpha * real_images + (1 - alpha) * fake_images

    with tf.GradientTape() as tape:
        tape.watch(interpolated_samples)
        interpolated_predictions = discriminator(interpolated_samples)

    gradients = tape.gradient(interpolated_predictions, interpolated_samples)
    gradients_sqr = tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3])
    gradient_penalty = tf.reduce_mean((gradients_sqr - 1.0) ** 2)

    return gradient_penalty

def wasserstein_loss(y_true, y_pred):
    """Calculates the Wasserstein loss for a sample batch.
    
    Args:
        y_true (tensor): Real images
        y_pred (tensor): Fake images
        
    Returns:
        tensor: Wasserstein loss
    """
    alpha = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0.0, maxval=1.0)
    interpolated_samples = alpha * real_imgs + (1 - alpha) * fake_imgs

    with tf.GradientTape() as tape:
        tape.watch(interpolated_samples)
        interpolated_predictions = discriminator(interpolated_samples)

    gradients = tape.gradient(interpolated_predictions, interpolated_samples)
    gradients_sqr = tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3])
    gradient_penalty = tf.reduce_mean((gradients_sqr - 1.0) ** 2)

    gradient_penalty_weight = 5.0

    w_loss = tf.reduce_mean(y_true * y_pred)

    w_loss += gradient_penalty_weight * gradient_penalty

    return w_loss

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
    model.add(Dropout(0.3))
    model.add(Dense(12544, activation=activation, kernel_regularizer=l2(l2_reg)))

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
    model = Sequential()

    model.add(Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding='same', input_shape=input_shape, kernel_regularizer=l2(l2_reg), name='conv1'))
    model.add(tf.keras.layers.Activation('relu'))
    
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(Conv2D(128, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_regularizer=l2(l2_reg), name='conv2'))
    model.add(tf.keras.layers.Activation('relu'))

    model.add(Flatten(name='flatten'))

    model.add(Dense(1, activation='linear', kernel_regularizer=l2(l2_reg), name='dense'))

    return model

def save_generated_images(epoch, generator, batch, examples=15, random_dim=1024):
    """Saves generated images to a file.
    
    Args:
        epoch (integer): Epoch number
        generator (Sequential): Generator model
        batch (integer): Batch number
        examples (integer): Number of examples
        random_dim (integer): Dimension of the latent vector
    """
    noise = np.random.normal(0, 1, (examples, random_dim))

    generated_images = generator.predict(noise)

    generated_images = (generated_images + 1) * 127.5
    generated_images = generated_images.astype(np.uint8)
    
    if not os.path.exists(f"{output_images_dir}/generated_images_{epoch}"):
        os.makedirs(f"{output_images_dir}/generated_images_{epoch}")

    for i in range(examples):
        image = generated_images[i, :, :, :]
        image = np.reshape(image, [28, 28])
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        
    plt.savefig(f"{output_images_dir}/generated_images_{epoch}/generated_image_{batch}.png")
        
if __name__ == '__main__':
    l2_reg = 2.5e-5
    activation = LeakyReLU(alpha=0.2)
    learning_rate = 0.0001
    beta1 = 0.5

    random_dim = 1024 
    input_shape = (28, 28, 1)

    generator = build_generator(random_dim)
    discriminator = build_discriminator(input_shape)

    generator.compile(loss=wasserstein_loss, optimizer=Adam(learning_rate, beta_1=beta1))
    discriminator.compile(loss=wasserstein_loss, optimizer=Adam(learning_rate, beta_1=beta1), metrics=['accuracy'])

    discriminator.trainable = False 

    gan_input = tf.keras.Input(shape=(random_dim,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = tf.keras.Model(gan_input, gan_output)

    gan.compile(loss=wasserstein_loss, optimizer=Adam(learning_rate, beta_1=beta1))

    (x_train, _), (_, _) = load_data_npz()
    x_train = np.expand_dims(x_train, axis=-1)
    x_train = (x_train / 255.0) * 2.0 - 1.0
    
    batch_size = 128
    epochs = 50
    sample_interval = 5
    
    discriminator_loss = []
    generator_loss = []

    for epoch in range(epochs + 1):
        for batch in range(x_train.shape[0] // batch_size):
            for _ in range(5):
                idx = np.random.randint(0, x_train.shape[0], batch_size)
                real_imgs = x_train[idx]

                noise = np.random.normal(0, 1, (batch_size, random_dim))
                fake_imgs = generator.predict(noise)

                discriminator.trainable = True

                valid = np.ones((batch_size, 1))
                fake = -np.ones((batch_size, 1))

                d_loss_real = discriminator.train_on_batch(real_imgs, valid)
                d_loss_fake = discriminator.train_on_batch(fake_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                gp = compute_gradient_penalty(real_imgs, fake_imgs)

                d_loss += gp
                discriminator.trainable = False

            noise = np.random.normal(0, 1, (batch_size, random_dim))

            g_loss = gan.train_on_batch(noise, np.ones(batch_size))

            if batch == 0:
                print(f'Epoch {epoch}/{epochs} | Batch {batch}/{x_train.shape[0] // batch_size} | D loss: {np.mean(d_loss):.4f} | G loss: {np.mean(g_loss):.4f}')
                save_generated_images(epoch, generator, batch)
                generator.save(f"{models_dir}/generator_model{epoch}.keras")
                
    # Salvar o loss em json
    with open(f"{loss_dir}/discriminator_loss.json", "w") as f:
        json.dump(discriminator_loss, f)
        
    with open(f"{loss_dir}/generator_loss.json", "w") as f:
        json.dump(generator_loss, f)