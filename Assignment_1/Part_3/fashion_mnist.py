import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

def load_data(normalize=True, one_hot=True):
    # Load the data
    train = pd.read_csv('Data/fashion-mnist_train.csv')
    test = pd.read_csv('Data/fashion-mnist_test.csv')

    # Split the data into X and y
    X_train = train.drop('label', axis=1)
    y_train = train['label']

    X_test = test.drop('label', axis=1)
    y_test = test['label']

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Normalize the data
    if normalize:
        X_train = X_train / 255
        X_test = X_test / 255

    # One-hot encode the labels
    if one_hot:
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train = X_train.reshape((-1, 784))
    X_test = X_test.reshape((-1, 784))

    # Train Validation Split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    return X_train, y_train, X_val, y_val, X_test, y_test

def plot_images(X, n=10):
    # Plot n images
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(X[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()