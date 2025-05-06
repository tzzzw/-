import pickle
import numpy as np
from tensorflow.keras.datasets import cifar10

def load_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = np.transpose(x_train, (0, 3, 1, 2)).astype(np.float32) / 255.0
    x_test = np.transpose(x_test, (0, 3, 1, 2)).astype(np.float32) / 255.0

    # 标准化处理（推荐）
    mean = np.mean(x_train, axis=(0, 2, 3), keepdims=True)
    std = np.std(x_train, axis=(0, 2, 3), keepdims=True) + 1e-7
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    return x_train, y_train, x_test, y_test
