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

class AffineTransformationLayer:
    def __init__(self, input_size, output_size, weight_initializer, optimizer=None, optimizer_params=None):
        self.input_size = input_size
        self.output_size = output_size
        if weight_initializer == 'zeros':
            self.weights = np.zeros((input_size, output_size))
            self.bias = np.zeros((1, output_size))
        elif weight_initializer == 'normal_init':
            self.weights = np.random.normal(0, 0.1, (input_size, output_size))
            self.bias = np.random.normal(0, 0.1, (1, output_size))
        if optimizer == 'momentum':
            self.vt_weights = np.zeros((input_size, output_size))
            self.vt_bias = np.zeros((1, output_size))
        if optimizer == 'nestrov_accelerated_gradient':
            self.vt_weights = np.zeros((input_size, output_size))
            self.vt_bias = np.zeros((1, output_size))
        if optimizer == 'adagrad':
            self.weight_gradient_cache = np.zeros((input_size, output_size))
            self.bias_gradient_cache = np.zeros((1, output_size))

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.bias

    def backward(self, output_error, learning_rate, optimizer=None, optimizer_params=None):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        if optimizer == None:
            self.weights -= learning_rate * weights_error
            self.bias -= learning_rate * output_error
        if optimizer == 'momentum':
            gamma = optimizer_params['gamma']
            self.vt_weights = gamma * self.vt_weights + learning_rate * weights_error
            self.weights -= self.vt_weights
            self.vt_bias = gamma * self.vt_bias + learning_rate * output_error
            self.bias -= self.vt_bias
        if optimizer == 'nestrov_accelerated_gradient':
            gamma = optimizer_params['gamma']
            vt_weights_prev = self.vt_weights
            vt_bias_prev = self.vt_bias
            self.vt_weights = gamma * self.vt_weights - learning_rate * weights_error
            self.vt_bias = gamma * self.vt_bias - learning_rate * output_error
            self.weights += -gamma * vt_weights_prev + (1 + gamma) * self.vt_weights
            self.bias += -gamma * vt_bias_prev + (1 + gamma) * self.vt_bias
        if optimizer == 'adagrad':
            epsilon = optimizer_params['epsilon']
            self.weight_gradient_cache += weights_error ** 2
            self.bias_gradient_cache += output_error ** 2
            self.weights -= learning_rate * weights_error / (np.sqrt(self.weight_gradient_cache) + epsilon)
            self.bias -= learning_rate * output_error / (np.sqrt(self.bias_gradient_cache) + epsilon)
        return input_error

class NonLinearTransformationLayer:
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
    
    def forward(self, input):
        self.input = input
        return self.activation(input)
    
    def backward(self, output_error, learning_rate, optimizer=None, optimizer_params=None):
        return output_error * self.activation_prime(self.input)

class ShapeHandler:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def forward(self, input):
        return np.reshape(input, (1, -1))
    
    def backward(self, output_error, learning_rate, optimizer=None, optimizer_params=None):
        return np.reshape(output_error, self.input_shape)

class SoftmaxLayer:
    def __init__(self, input_size):
        self.input_size = input_size
    
    def forward(self, input):
        self.input = input
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output
    
    def backward(self, output_error, learning_rate, optimizer=None, optimizer_params=None):
        _ = np.zeros(output_error.shape)
        out = np.tile(self.output.T, self.input_size)
        return self.output * np.dot(output_error, np.identity(self.input_size) - out)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(x):
    return np.exp(-x) / (1 + np.exp(-x))**2

def relu(x):
    return np.maximum(x, 0)

def relu_prime(x):
    return np.array(x >= 0).astype('int')

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_pred.size

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

class NeuralNetwork:
    def __init__(self, n_inputs,  input_shape, hidden_layer_sizes, n_outs, activation_fn, activation_fn_derivative, weight_initialization, loss, loss_prime, optimizer, optimizer_params):
        self.network = []
        self.network.append(ShapeHandler(input_shape))
        # Input To Hidden
        self.network.append(AffineTransformationLayer(n_inputs, hidden_layer_sizes[0], weight_initialization, optimizer, optimizer_params))
        self.network.append(NonLinearTransformationLayer(activation_fn, activation_fn_derivative))
        # Hidden To Hidden
        for i in range(len(hidden_layer_sizes) - 1):
            self.network.append(AffineTransformationLayer(hidden_layer_sizes[i], hidden_layer_sizes[i+1], weight_initialization, optimizer, optimizer_params))
            self.network.append(NonLinearTransformationLayer(activation_fn, activation_fn_derivative))
        # Hidden To Output
        self.network.append(AffineTransformationLayer(hidden_layer_sizes[-1], n_outs, weight_initialization, optimizer, optimizer_params))
        self.network.append(SoftmaxLayer(n_outs))
        self.loss = loss
        self.loss_prime = loss_prime
        self.train_loss = []
        self.val_loss = []
        self.train_acc = []
        self.val_acc = []
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
    
    def predict(self, x):
        for layer in self.network:
            x = layer.forward(x)
        return x
    
    def fit(self, x_train, y_train, x_valid, y_valid, epochs=1000, learning_rate=0.1, batch_size=32, early_stopping=True, patience=10, verbose=True):
        x_batches_train = np.array_split(x_train, len(x_train)/ batch_size)
        y_batches_train = np.array_split(y_train, len(y_train)/ batch_size)
        for epoch in range(epochs):
            error = 0
            error_val = 0
            for x_train, y_train in zip(x_batches_train,y_batches_train):
                for x, y_true in zip(x_train, y_train):

                    output = x
                    
                    output = self.forward(output)

                    error += self.loss(y_true, output)

                    output_error = self.loss_prime(y_true, output)

                    self.backward(output_error, learning_rate)

                    error /= len(x_train)

            error_val = sum([self.loss(y, self.predict(x)) for x, y in zip(x_valid, y_valid)]) / len(x_valid) 

            self.train_loss.append(error)
            self.val_loss.append(error_val)
            train_acc = 0
            for x, y in zip(x_train, y_train):
                train_acc += accuracy(np.argmax(y), np.argmax(self.predict(x)))
            self.train_acc.append(train_acc / len(x_train))
            val_acc = 0
            for x, y in zip(x_valid, y_valid):
                val_acc += accuracy(np.argmax(y), np.argmax(self.predict(x)))
            self.val_acc.append(val_acc / len(x_valid))

            if early_stopping:
                if epoch > patience:
                    if self.train_loss[-1] > np.mean(self.val_loss[-(patience+1):-1]):
                        print('Early Stopping')
                        break
            if verbose:
                if epoch % 1 == 0:
                    print(f"Epoch: {epoch}, Training Loss: {error}, Validation Loss: {error_val}, Training Accuracy: {self.train_acc[-1]}, Validation Accuracy: {self.val_acc[-1]}")

    def forward(self, x):
        for layer in self.network:
            x = layer.forward(x)
        return x

    def backward(self, output_error, learning_rate):
        for layer in reversed(self.network):
            output_error = layer.backward(output_error, learning_rate, self.optimizer, self.optimizer_params)

    def predict_proba(self, x):
        return self.predict(x)

    def score(self, x, y):
        cnt = 0
        for x_v, y_v in zip(x,y):
            op = self.predict(x_v)
            if (np.argmax(op) == np.argmax(y_v)):
                cnt+=1
        return cnt/y.shape[0]

def save_model(model, filename):
    import pickle
    with open(f"Model_Saves/{filename}", 'wb') as f:
        pickle.dump(model, f)

def load_model(filename):
    import pickle
    with open(f"Model_Saves/{filename}", 'rb') as f:
        return pickle.load(f)