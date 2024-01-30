import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

class Critic:
    def __init__(self, learning_rate, beta1, l2_reg, activation, input_shape, gradient_penalty_weight):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.l2_reg = l2_reg
        self.activation = activation
        self.input_shape = input_shape
        self._optimizer = optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta1)
        self.gradient_penalty_weight = gradient_penalty_weight
        self._model = self.build_model()
        
    def build_model(self):
        model = Sequential()

        # Primeira camada convolucional com regularização L2
        model.add(Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding='same', input_shape=self.input_shape, kernel_regularizer=l2(self.l2_reg), name='conv1'))
        model.add(tf.keras.layers.Activation('relu'))

        # Segunda camada convolucional com regularização L2
        model.add(Conv2D(128, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_regularizer=l2(self.l2_reg), name='conv2'))
        model.add(tf.keras.layers.Activation('relu'))

        # Camada de flatten
        model.add(Flatten(name='flatten'))

        # Camada totalmente conectada (dense) com regularização L2
        model.add(Dense(1, activation='linear', kernel_regularizer=l2(self.l2_reg), name='dense'))

        return model
    
    def get_model(self):
        return self._model
    
    def loss(self, real_output, fake_output, gradient_penalty, gradient_penalty_weight):
        total_loss = (-1 * tf.reduce_mean(real_output) + tf.reduce_mean(fake_output) + gradient_penalty_weight * gradient_penalty)
        return total_loss
    
    def optimizer(self):
        return self._optimizer
    
    def save_model(self, path):
        self._model.save(path)