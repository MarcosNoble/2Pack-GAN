import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Conv2DTranspose, Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

class Generator:
    def __init__(self, learning_rate, beta1, l2_reg, activation, random_dim):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.l2_reg = l2_reg
        self.activation = activation
        self.random_dim = random_dim
        self._optimizer = Adam(learning_rate=self.learning_rate, beta_1=self.beta1)
        self.model = self.build_model()
        
    def build_model(self):
        model = Sequential()

        # Camadas totalmente conectadas com regularização L2
        model.add(Dense(64, input_dim=self.random_dim, activation=self.activation, kernel_regularizer=l2(self.l2_reg)))
        model.add(Dense(1024, activation=self.activation, kernel_regularizer=l2(self.l2_reg)))
        model.add(Dense(12544, activation=self.activation, kernel_regularizer=l2(self.l2_reg)))

        # Reshape para 7x7x256
        model.add(Reshape((7, 7, 256)))

        # Deconvolution para 14x14x64
        model.add(Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', activation=self.activation, kernel_regularizer=l2(self.l2_reg)))

        # Deconvolution para 28x28x32
        model.add(Conv2DTranspose(32, (4, 4), strides=(2, 2), padding='same', activation=self.activation, kernel_regularizer=l2(self.l2_reg)))

        # Camada de saída (convolução para 28x28)
        model.add(Conv2D(1, (1, 1), padding='same', activation='tanh'))

        return model
    
    def get_model(self):
        return self.model
    
    def loss(self, fake_output):
        return -1 * tf.reduce_mean(fake_output)
    
    def optimizer(self):
        return self._optimizer
    
    def save_model(self, path):
        self.model.save(path)