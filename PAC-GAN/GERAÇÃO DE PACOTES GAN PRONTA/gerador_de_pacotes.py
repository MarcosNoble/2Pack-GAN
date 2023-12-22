import numpy as np
import os
import tensorflow as tf
from keras.models import load_model

current_dir = os.path.dirname(os.path.abspath(__file__))

generated_packets_dir = os.path.join(current_dir, 'generated_packets')


if not os.path.exists(generated_packets_dir):
    os.makedirs(generated_packets_dir)

def wasserstein_loss(y_true, y_pred):
    """Calculates the Wasserstein loss for a sample batch.

    Args:
        y_true (tensor): Real images
        y_pred (tensor): Fake images

    Returns:
        tensor: Wasserstein loss
    """
    # Amostras interpoladas
    alpha = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0.0, maxval=1.0)
    interpolated_samples = alpha * real_imgs + (1 - alpha) * fake_imgs

    with tf.GradientTape() as tape:
        tape.watch(interpolated_samples)
        interpolated_predictions = discriminator(interpolated_samples)

    gradients = tape.gradient(interpolated_predictions, interpolated_samples)
    gradients_sqr = tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3])
    gradient_penalty = tf.reduce_mean((gradients_sqr - 1.0) ** 2)

    # Parâmetro de penalidade do gradiente
    gradient_penalty_weight = 10.0

    # Calcula a perda de Wasserstein padrão
    w_loss = tf.reduce_mean(y_true * y_pred)

    # Adiciona a penalização do gradiente à perda de Wasserstein
    w_loss += gradient_penalty_weight * gradient_penalty

    return w_loss


def generate_packets_by_gan(num_images):
    """Generate packets using a GAN
    
    Args:
        num_images (integer): Number of images to generate
    """
    generator_path = os.path.join(current_dir, 'models/generators/generator_model50.keras')

    # Carregar o modelo gerador
    generator = load_model(generator_path, custom_objects={'wasserstein_loss': wasserstein_loss})

    # Gerar ruído aleatório para entrada do gerador
    noise = np.random.normal(0, 1, (num_images, 1024))  # Substitua input_dim pelo tamanho adequado do ruído

    # Gerar imagens usando o gerador
    generated_images = generator.predict(noise)

    generated_packets = {"generated_packets": []}

    # Salvar as imagens geradas em um arquivo npz, transformando elas para um intervalo entre 0 e 255
    for image in generated_images:
        image = (image + 1) * 127.5
        image = image.astype(np.uint8)
    
        generated_packets["generated_packets"].append(image)
    
    np.savez(os.path.join(generated_packets, 'generated_packets'), **generated_packets)

