#!/usr/bin/env python3
"""Import numpy"""

import numpy as np
"""Import pickle"""

import pickle
"""class that represents a deep neural network"""


class DeepNeuralNetwork:
    """
    class that represents a deep neural network
    performing binary classification
    """
    def __init__(self, nx, layers):
        """
        class constructor
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) < 1:
            raise TypeError("layers must be a list of positive integers")

        weights = {}
        previous = nx

        for index, layer in enumerate(layers, 1):

            if type(layer) is not int or layer < 0:
                raise TypeError("layers must be a list of positive integers")

            weights["b{}".format(index)] = np.zeros((layer, 1))
            weights["W{}".format(index)] = (
                np.random.randn(layer, previous) * np.sqrt(2 / previous))
            previous = layer

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = weights

    @property
    def L(self):
        return (self.__L)

    @property
    def cache(self):
        return (self.__cache)

    @property
    def weights(self):
        return (self.__weights)

    def forward_prop(self, X):
        """
        forward propagation method for DeepNeuralNetwork class
        :param X: numpy.ndarray of shape (nx, m), input data
        :return: output of the neural network and the cache, respectively
        """
        self.cache['A0'] = X
        for l in range(1, self.L):
            self.cache['Z' + str(l)] = np.dot(self.weights['W' + str(l)], self.cache['A' + str(l - 1)]) + \
                                       self.weights['b' + str(l)]
            self.cache['A' + str(l)] = 1 / (1 + np.exp(-self.cache['Z' + str(l)]))

        self.cache['Z' + str(self.L)] = np.dot(self.weights['W' + str(self.L)], self.cache['A' + str(self.L - 1)]) + \
                                        self.weights['b' + str(self.L)]
        self.cache['A' + str(self.L)] = np.exp(self.cache['Z' + str(self.L)]) / np.sum(np.exp(self.cache['Z' + str(self.L)]),
                                                                                        axis=0)
        return self.cache['A' + str(self.L)], self.cache

    def cost(self, Y, A):
        """
        cost method for DeepNeuralNetwork class
        :param Y: numpy.ndarray of shape (classes, m), one-hot labels
        :param A: numpy.ndarray of shape (classes, m), probability output of the neural network
        :return: cross-entropy cost
        """
        return -np.sum(Y * np.log(A))

    def evaluate(self, X, Y):
        """
        evaluation method for DeepNeuralNetwork class
        :param X: numpy.ndarray of shape (nx, m), input data
        :param Y: numpy.ndarray of shape (classes, m), one-hot labels
        :return: the neuron’s prediction and the cost of the network
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.argmax(A, axis=0)
        Y_hat = np.argmax(Y, axis=0)
        accuracy = np.sum(prediction == Y_hat) / Y.shape[1]
        return prediction, accuracy, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        calculates gradient descent
        """
        m = Y.shape[1]
        back = {}

        for index in range(self.L, 0, -1):

            A = cache["A{}".format(index - 1)]

            # checks if the current layer is the output layer
            if index == self.L:
                # the derivative of the cost with respect
                # to the output activations A
                # is computed as A - Y
                back["dz{}".format(index)] = (cache["A{}".format(index)] - Y)
            else:
                # compute derivative w.r.t the activations
                # of the previous layer
                # retrieve  derivative w.r.t the activations
                # of the current layer+1
                dz_prev = back["dz{}".format(index + 1)]
                # retrieve the activations of the current layer
                A_current = cache["A{}".format(index)]
                # compute the derivative of the cost with
                # respect to the activations
                back["dz{}".format(index)] = (
                    np.matmul(W_prev.transpose(), dz_prev) *
                    (A_current * (1 - A_current)))

            # compute the gradients of the weights
            # and biases of a layer during backpropagation
            # dz is the error of the current layer
            dz = back["dz{}".format(index)]
            # dW is the gradient of the weights
            dW = (1 / m) * (np.matmul(dz, A.transpose()))
            # db is the gradient of the biases, along the m axis
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

            W_prev = self.weights["W{}".format(index)]
            self.__weights["W{}".format(index)] = (
                self.weights["W{}".format(index)] - (alpha * dW))

            self.__weights["b{}".format(index)] = (
                self.weights["b{}".format(index)] - (alpha * db))

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Train a deep neural network
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        if graph:
            import matplotlib.pyplot as plt
            x_points = np.arange(0, iterations + 1, step)
            points = []

        for itr in range(iterations):
            A, cache = self.forward_prop(X)
            if verbose and (itr % step) == 0:
                cost = self.cost(Y, A)
                print("Cost after " + str(itr) + " iterations: " + str(cost))
            if graph and (itr % step) == 0:
                cost = self.cost(Y, A)
                points.append(cost)
            self.gradient_descent(Y, cache, alpha)

        itr += 1

        if verbose:
            cost = self.cost(Y, A)
            print("Cost after " + str(itr) + " iterations: " + str(cost))

        if graph:
            cost = self.cost(Y, A)
            points.append(cost)
            y_points = np.asarray(points)
            plt.plot(x_points, y_points, 'b')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return (self.evaluate(X, Y))

    def save(self, filename):
        """
        Saves the instance object to a file in pickle format
        """
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Loads a pickled DeepNeuralNetwork object
        """
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
