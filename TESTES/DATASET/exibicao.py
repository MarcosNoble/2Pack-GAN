import matplotlib.pyplot as plt
import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

# Caminho para o arquivo npz
file_path = os.path.join(current_dir, 'dataset_ip.npz')

# Exibir uma imagem do dataset
dataset = np.load(file_path)

imagem = dataset['x_train'][0]

imagem[2][20:] = 300
imagem[3][20:] = 300

imagem[4][0:16] = 300
imagem[5][0:16] = 300

# Exibir uma imagem do dataset em rgb
plt.imshow(imagem)
plt.show()

