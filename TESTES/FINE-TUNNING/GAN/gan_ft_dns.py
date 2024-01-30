from distutils.command import build
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Conv2DTranspose, Conv2D, Flatten, LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from data_loader import load_data_npz

current_dir = os.path.dirname(os.path.abspath(__file__))
output_images_dir = os.path.join(current_dir, 'output_images_ft_dns')

models_dir = os.path.join(current_dir, 'models')

models_ft_dir = os.path.join(current_dir, 'models_ft_dns')

        
if not os.path.exists(f"{models_ft_dir}"):
        os.makedirs(f"{models_ft_dir}")
        
# Função de regularização L2
l2_reg = 2.5e-5

# Função de ativação Leaky ReLU
activation = LeakyReLU(alpha=0.2)

# Taxa de aprendizado do otimizador Adam
learning_rate = 0.001

# Beta1 do otimizador Adam (taxa de decaimento exponencial)
beta1 = 0.5

def compute_gradient_penalty(real_images, fake_images):
    """Calculates the gradient penalty for a batch of "real" and "fake" images, as a loss function for the discriminator.
    
    Args:
        real_images (tensor): Real images
        fake_images (tensor): Fake images
        
    Returns:
        tensor: Gradient penalty
    """
    # Amostras interpoladas
    alpha = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0.0, maxval=1.0)
    interpolated_samples = alpha * real_images + (1 - alpha) * fake_images

    with tf.GradientTape() as tape:
        tape.watch(interpolated_samples)
        interpolated_predictions = discriminator(interpolated_samples)

    gradients = tape.gradient(interpolated_predictions, interpolated_samples)
    gradients_sqr = tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3])
    gradient_penalty = tf.reduce_mean((gradients_sqr - 1.0) ** 2)

    return gradient_penalty

@tf.keras.utils.register_keras_serializable()
def wasserstein_loss(y_true, y_pred):
    """Calculates the Wasserstein loss for a batch of "real" and "fake" images, as a loss function for the discriminator.
    
    Args:
        y_true (tensor): True labels
        y_pred (tensor): Predicted labels
        
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

# Gerador
def build_generator(random_dim):
    """Builds the generator model.
    
    Args:
        random_dim (integer): Dimension of the latent vector
        
    Returns:
        Sequential: Generator model
    """
    model = tf.keras.Sequential()

    # Camadas totalmente conectadas com regularização L2
    model.add(Dense(64, input_dim=random_dim, activation=activation, kernel_regularizer=l2(l2_reg)))
    model.add(Dense(1024, activation=activation, kernel_regularizer=l2(l2_reg)))
    model.add(Dense(12544, activation=activation, kernel_regularizer=l2(l2_reg)))

    # Reshape para 7x7x256
    model.add(Reshape((7, 7, 256)))

    # Deconvolution para 14x14x64
    model.add(Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', activation=activation, kernel_regularizer=l2(l2_reg)))

    # Deconvolution para 28x28x32
    model.add(Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same', activation=activation, kernel_regularizer=l2(l2_reg)))

    # Camada de saída (convolução para 28x28)
    model.add(Conv2D(1, (1, 1), padding='same', activation='tanh'))

    return model

# Discriminador
def build_discriminator(input_shape):
    """Builds the discriminator model.
    
    Args:
        input_shape (tuple): Shape of the input images
        
    Returns:
        Sequential: Discriminator model
    """
    model = Sequential()

    # Primeira camada convolucional com regularização L2
    model.add(Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding='same', input_shape=input_shape, kernel_regularizer=l2(l2_reg), name='conv1'))
    model.add(tf.keras.layers.Activation('relu'))

    # Segunda camada convolucional com regularização L2
    model.add(Conv2D(128, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_regularizer=l2(l2_reg), name='conv2'))
    model.add(tf.keras.layers.Activation('relu'))

    # Camada de flatten
    model.add(Flatten(name='flatten'))

    # Camada totalmente conectada (dense) com regularização L2
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

    # Reescalar para o intervalo [0, 255]
    generated_images = (generated_images + 1) * 127.5
    generated_images = generated_images.astype(np.uint8)
    
    if not os.path.exists(f"{output_images_dir}/generated_images_{epoch}"):
        os.makedirs(f"{output_images_dir}/generated_images_{epoch}")

    for i in range(examples):
        # Juntar todas as imagens geradas em uma única imagem
        image = generated_images[i, :, :, :]
        image = np.reshape(image, [28, 28])
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        
    # Salvar imagem
    plt.savefig(f"{output_images_dir}/generated_images_{epoch}/generated_image_{batch}.png")
        

if __name__ == '__main__':
    # Construir o gerador e o discriminador
    random_dim = 1024  # Dimensão do vetor latente
    input_shape = (28, 28, 1)  # Dimensões das imagens de entrada

    generator = build_generator(random_dim)
    discriminator = build_discriminator(input_shape)
    
    generator.load_weights(f"{models_dir}/generator50.weights.h5")
    discriminator.load_weights(f"{models_dir}/discriminator50.weights.h5")
    
    # Especificar o número de camadas que deseja congelar no discriminador
    '''n_layers_to_freeze_discriminator = 2  # Ajuste conforme necessário

    # Congelar os pesos das camadas iniciais do discriminador
    for layer in discriminator.layers[:n_layers_to_freeze_discriminator]:
        layer.trainable = False'''
        
    #discriminator.summary()
        
    # Especificar o número de camadas que deseja congelar no gerador
    '''n_layers_to_freeze_generator = 2  # Ajuste conforme necessário

    # Congelar os pesos das camadas iniciais do gerador
    for layer in generator.layers[:n_layers_to_freeze_generator]:
        layer.trainable = False'''

    # Compilar o modelo do gerador
    generator.compile(loss=wasserstein_loss, optimizer=Adam(learning_rate, beta_1=beta1))

    # Compilar o modelo do discriminador
    discriminator.compile(loss=wasserstein_loss, optimizer=Adam(learning_rate, beta_1=beta1), metrics=['accuracy'])

    # Combinação do gerador e do discriminador em um modelo GAN
    discriminator.trainable = False  # Congela os pesos do discriminador durante o treinamento do GAN
    
    #discriminator.summary()

    gan_input = tf.keras.Input(shape=(random_dim,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = tf.keras.Model(gan_input, gan_output)

    # Compilar o modelo GAN
    gan.compile(loss=wasserstein_loss, optimizer=Adam(learning_rate, beta_1=beta1))
    
    # Configurações de treinamento
    batch_size = 64
    epochs = 50 # Número de épocas
    sample_interval = 100  # Intervalo para salvar e exibir imagens geradas

    # Loop de treinamento
    import numpy as np
    import matplotlib.pyplot as plt

    # Carregar o conjunto de dados
    (x_train, _), (_, _) = load_data_npz()

    # Expandir dimensões
    x_train = np.expand_dims(x_train, axis=-1)

    # Normalizar para o intervalo [-1, 1]
    x_train = (x_train / 255.0) * 2.0 - 1.0
    
    # Loop de treinamento:
    for epoch in range(epochs + 1):
        for batch in range(x_train.shape[0] // batch_size):
            # Treinar o discriminador
            for _ in range(5):
                # Amostras reais
                idx = np.random.randint(0, x_train.shape[0], batch_size)
                real_imgs = x_train[idx]

                # Amostras falsas
                noise = np.random.normal(0, 1, (batch_size, random_dim))
                fake_imgs = generator.predict(noise)

                #for layer in generator.layers[n_layers_to_freeze_generator:]:
                #    layer.trainable = True
                discriminator.trainable = True
                
                #discriminator.summary()

                # Adversarial ground truths
                valid = np.ones((batch_size, 1))
                fake = -np.ones((batch_size, 1))

                d_loss_real = discriminator.train_on_batch(real_imgs, valid)
                d_loss_fake = discriminator.train_on_batch(fake_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # Penalidade do gradiente
                gp = compute_gradient_penalty(real_imgs, fake_imgs)

                # Atualizar pesos do discriminador
                d_loss += gp
                #for layer in generator.layers[:n_layers_to_freeze_generator]:
                    #layer.trainable = False
                discriminator.trainable = False

            # Treinar o gerador
            noise = np.random.normal(0, 1, (batch_size, random_dim))

            # Rótulos indicam que as imagens geradas são "reais"
            g_loss = gan.train_on_batch(noise, np.ones(batch_size))

            if batch % sample_interval == 0:
                # Exibir o progresso
                print(f'Epoch {epoch}/{epochs} | Batch {batch}/{x_train.shape[0] // batch_size} | D loss: {np.mean(d_loss):.4f} | G loss: {np.mean(g_loss):.4f}')
                save_generated_images(epoch, generator, batch)
                generator.save(f"{models_ft_dir}/generator_model{epoch}.keras")
            
