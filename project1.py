import sys
import numpy as np

class Neuron:
    def __init__(self, num_inputs, activation = 'logistic',  \
                learning_rate = 0.01, w_0 = None):
        '''
        Arguments:
        ----------
        num_inputs: int
            - 
        activation: string, optional
            - Default: 'logistic'
            - Options: 'logistic' or 'linear'
            - Specifies the activation to use for this given neuron
        learning_rate: float, optional
            - Default: 0.01
            - Learning rate used in gradient descent optimization of the weights
        w_0: array of size (num_inputs,), optional
            - Default: None
            - Initial weights for the neuron
            - If None, randomly initializes the weights
        '''
        self.n = num_inputs

        # Set the activation function based on user input
        # Also sets activation derivative as lambda function
        if activation == 'logistic':
            self.activate = lambda x : (1 / (1 + np.exp(-x)))
            self.activationderivative = \
                lambda x: self.activate(x) * (1 - self.activate(x))
        elif activation == 'linear':
            self.activate = lambda x : x
            self.activationderivative = lambda x: 1

        self.alpha = learning_rate # Learning rate for GD

        # If no weights specified:
        if w_0 == None:
            # Set to random (uniform distribution between 0 and 1)
            self.w = np.random.rand(num_inputs + 1) # Add 1 for bias
            pass

    def calculate(self, x):
        # Formula: W*x + b (W is weights vector, b is bias)
        return np.dot(self.w[0:self.n], x) + self.w[-1]

    def calcpartialderivative(self):
        pass

    def updateweights(self):
        pass