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
        '''
        Calculates feedforward input through the neuron
        Arguments:
        ----------
        x: (n, ) array-like
            - Inputs to the neuron
            - n must match the number of inputs

        Returns:
        --------
        output: (n + 1, ) array-like
            - Calculated output
            - n + 1 for output size due to bias
        '''
        # Saves the input to the neuron
        self.input = x

        # Formula: W*x + b (W is weights vector, b is bias)
        # Saves the output to the Neuron
        self.output = np.dot(self.w[0:self.n], x) + self.w[-1]

        return self.activate(self.output) # Returns the output

    def calcpartialderivative(self, l1_deltas):
        '''
        Calculates and stores partial derivatives for each weight in this neuron
        Arguments:
        ----------
        l1_deltas: list
            - List of delta's in the (l+1) layer (given that this 
                is the l-th layer)
            - Will be multiplied by w's inside this function

        Returns:
        --------
        delta: float
            - Calculated delta for this neuron
            - Only the delta value, not multiplied by anything
        '''
        # Intialize partial derivative array:
        dE_dw = np.zeros(self.n + 1)
        
        # Delta = dE/dOo1 * dOo1/dneto1
        dE_dO = sum([l1_deltas[i] * self.w[i] for i in range(0, self.n)])
            # Summing all delta * w_i terms to account for partial of error

        dO_dnet = self.activationderivative(self.output) 
            # Derviative of activation function, plugging in self.input

        # Delta for this variable
        delta = dE_dO * dO_dnet

        # Iterate over weights:
        for i in range(self.n):
            dE_dw[i] = delta * self.input[i]

        # Derivative for bias:
        dE_dw[-1] = delta # It's just delta here

        # Stores the vector of partial derivatives internally
        self.dE_dw = dE_dw

        return delta

    def updateweights(self):
        '''
        Changes the weights
        Arguments:
        ----------
        No arguments, member method

        Returns:
        --------
        No return, only updates internals of class
        '''
        # Iterate over the weights and update based on 
        for i in range(self.n + 1):
            self.w[i] = self.w[i] - self.alpha * self.dE_dw[i]

class FullyConnectedLayer:
    def __init__(self, num_neurons, num_inputs, learning_rate=0.01, \
        activation='logistic', w_0=None):

        # Set numbers for layers:
        self.n_i = num_inputs
        self.n_n = num_neurons

        # Set up the neurons:
        self.neurons = []

        # Creates an (n_n,) array of neurons
        for i in range(0, self.n_n):
            # Set up each neuron with desired properties
            new_neuron = Neuron(num_inputs = self.n_i,
                                activation = activation,
                                learning_rate = learning_rate,
                                w_0 = w_0)
            # Add this neuron to the list:
            self.neurons.append(new_neuron)
            

    def calculate(self, x):
        '''
        Calculates vector of outputs from neurons in layer
        Arguments:
        ----------
        x: array-like
            - Input to the layer from l - 1 layer

        Returns:
        --------
        output: list
            - List of outputs from each neuron in layer
        '''
        output = []

        # Iterate over all neurons
        for i in range(0, self.n_n):
            # Calculate each neuron's output
            #   Note: this updates the internals of each neuron
            output.append(self.neurons[i].calculate(x))

        return output

    def calculatewdeltas(self, w_delta):
        ''' 
        Calculates list of deltas for each neuron in layer
            - Performs updates on the weights in the layer for each neuron
        Arguments:
        ----------
        w_delta: array-like
            - deltas from layer (l+1)

        Returns:
        --------
        all_deltas: list
            - delta values for all neurons in layer
        ''' 
        all_deltas = []

        # Iterate over all neurons:
        for i in range(0, self.n_n):
            # Calculate delta value for this specific neuron:
            delta_i = self.neurons[i].calcpartialderivative(w_delta)

            # Add delta to the list
            all_deltas.append(delta_i)
            # Update the weights in this neuron
            self.neurons[i].updateweights()
        
        return all_deltas

class NeuralNetwork:
    def __init__(self, num_layers, num_neurons, num_inputs, lr=0.01, loss_func=None, \
        act_funcs=None):
        pass
