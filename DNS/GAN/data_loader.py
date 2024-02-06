import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
dataset_dir = os.path.join(parent_dir, 'DATASET')
file_path = os.path.join(dataset_dir, 'dataset_dns.npz')

def load_data_npz():
        """Loads the dataset from a NPZ file

        Returns:
                tuple: Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
        """
        with np.load(file_path, allow_pickle=True) as f:
                x_train, y_train = f["x_train"], f["y_train"]
                x_test, y_test = f["x_test"], f["y_test"]
                print("x_train shape:", x_train.shape)
                print("y_train shape:", y_train.shape)

        return (x_train, y_train), (x_test, y_test)