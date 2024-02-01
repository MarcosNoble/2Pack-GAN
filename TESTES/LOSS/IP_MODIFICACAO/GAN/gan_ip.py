from email.mime import image
import os
from tabnanny import check
from numpy import gradient
from sympy import plot
import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Conv2DTranspose, Conv2D, Flatten, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from torch import ge, seed
from data_loader import load_data_npz
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def generate_and_save_images(model, test_input, epoch, img_path):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))
    
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
        
    plt.savefig(f"{img_path}/generated_images_{epoch}.png")
    
        
@tf.function
def critic_train_step(generator, critic, real_input_batch, random_dim):
    noise = tf.random.normal([batch_size, random_dim])
    
    with tf.GradientTape() as critic_tape:
        with tf.GradientTape() as gp_tape:
            generated_images = generator.get_model()(noise, training=True)
            
            epsilon = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0, dtype=tf.float64)
            real_input_batch = tf.cast(real_input_batch, tf.float64)
            generated_images = tf.cast(generated_images, tf.float64)
            mixed_outputs = real_input_batch * epsilon + generated_images * (1 - epsilon)
            mixed_predictions = critic.get_model()(mixed_outputs, training=True)
            
        grad = gp_tape.gradient(mixed_predictions, [mixed_outputs])[0]
        grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3]))
        gradient_penalty = tf.reduce_mean(tf.square(grad_norm - 1))
        
        gradient_penalty = tf.cast(gradient_penalty, tf.float32)
        
        critic_real_output = critic.get_model()(real_input_batch, training=True)
        critic_fake_output = critic.get_model()(generated_images, training=True)
        
        critic_loss = critic.loss(critic_real_output, critic_fake_output, gradient_penalty, gradient_penalty_weight)
        
    critic_gradients = critic_tape.gradient(critic_loss, critic.get_model().trainable_variables)
    critic.optimizer().apply_gradients(zip(critic_gradients, critic.get_model().trainable_variables))
    
    return critic_loss

@tf.function
def generator_train_step(generator, critic, batch_size, random_dim):
    noise = tf.random.normal([batch_size, random_dim])
    
    with tf.GradientTape() as gen_tape:
        generated_images = generator.get_model()(noise, training=True)
        critic_fake_output = critic.get_model()(generated_images, training=True)
        gen_loss = generator.loss(critic_fake_output)
        
    gen_gradients = gen_tape.gradient(gen_loss, generator.get_model().trainable_variables)
    generator.optimizer().apply_gradients(zip(gen_gradients, generator.get_model().trainable_variables))
    
    return gen_loss
            
def plot_loss(generator_loss_history, critic_loss_history):
    # Gerar um gráfico com o histórico de perda do gerador e do discriminador
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Critic Loss During Training")
    plt.plot(generator_loss_history, label="Generator")
    plt.plot(critic_loss_history, label="Critic")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{current_dir}/loss.png")

def train(generator, critic, output_images_dir, epochs, num_generate, random_dim, critic_iterations, batch_size, x_train):
    tqdm.write("\n-------- Starting training --------\n")
    
    generator_loss_history = []
    critic_loss_history = []
    
    seed = tf.random.normal([num_generate, random_dim])
    
    
    for epoch in range(epochs):
        tqdm.write('Epoch: {}/{}'.format(epoch + 1, epochs))
        
        for batch in tqdm(range(x_train.shape[0] // batch_size)):
            image_batch = x_train[batch * batch_size:(batch + 1) * batch_size]
            temp_loss = []
            
            for _ in range(critic_iterations):
                temp_loss.append(critic_train_step(generator, critic, image_batch, random_dim))
            critic_loss_history.append(tf.reduce_mean(temp_loss))
            generator_loss_history.append(generator_train_step(generator, critic, batch_size, random_dim))
            
                
        generator.save_model(f"{models_dir}/generator_{epoch + 1}.keras")
        #critic.save_model(f"{models_dir}/critic_{epoch + 1}.keras")
        
        generate_and_save_images(generator.get_model(), seed, epoch + 1, output_images_dir)
        
        tqdm.write('Generator loss: {}'.format(generator_loss_history[-1]))
        tqdm.write('Critic loss: {}'.format(critic_loss_history[-1]))
        
        plot_loss(generator_loss_history, critic_loss_history)
        
        generator_loss_history.clear()
        critic_loss_history.clear()
        
    tqdm.write("\n-------- Training finished --------\n")
                

if __name__ == '__main__':
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    output_images_dir = os.path.join(current_dir, 'output_images')
    
    if not os.path.exists(f"{output_images_dir}"):
        os.makedirs(f"{output_images_dir}")

    models_dir = os.path.join(current_dir, 'models')

    if not os.path.exists(f"{models_dir}"):
        os.makedirs(f"{models_dir}")
    
    # Função de regularização L2
    l2_reg = 2.5e-5

    # Função de ativação Leaky ReLU
    activation = LeakyReLU(alpha=0.2)

    # Taxa de aprendizado do otimizador Adam
    learning_rate = 0.001

    # Beta1 do otimizador Adam (taxa de decaimento exponencial)
    beta1 = 0.5

    gradient_penalty_weight = 10.0
    
    random_dim = 1024  # Dimensão do vetor latente
    input_shape = (28, 28, 1)  # Dimensões das imagens de entrada
    
    from generator import Generator
    from critic import Critic
    
    generator = Generator(learning_rate, beta1, l2_reg, activation, random_dim)
    critic = Critic(learning_rate, beta1, l2_reg, activation, input_shape, gradient_penalty_weight)
        
    # Configurações de treinamento
    batch_size = 64
    epochs = 300 # Número de épocas

    # Carregar o conjunto de dados
    (x_train, _), (_, _) = load_data_npz()

    # Expandir dimensões
    x_train = np.expand_dims(x_train, axis=-1)

    # Normalizar para o intervalo [-1, 1]
    x_train = (x_train / 255.0) * 2.0 - 1.0
    
    train(generator, critic, output_images_dir, epochs, 16, random_dim, 5, batch_size, x_train)