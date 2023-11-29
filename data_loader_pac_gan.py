import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'dataset.npz')

def load_data_npz():
    with np.load(file_path, allow_pickle=True) as f:
            x_train, y_train = f["x_train"], f["y_train"]
            x_test, y_test = f["x_test"], f["y_test"]

    return (x_train, y_train), (x_test, y_test)