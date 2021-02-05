import sys
import numpy as np

class Neuron:
    def __init__(self, num_inputs, w_0, activation = 'logistic',  \
                learning_rate = 0.01):
        '''
        Arguments:
        ----------
        num_inputs: int
            - 
        w_0: array of size (num_inputs,)
            - Initial weights for the neuron
            - If None, randomly initializes the weights
        activation: string, optional
            - Default: 'logistic'
            - Options: 'logistic' or 'linear'
            - Specifies the activation to use for this given neuron
        learning_rate: float, optional
            - Default: 0.01
            - Learning rate used in gradient descent optimization of the weights
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

        # Setting of random weights is done in FullyConnectedLayer
        self.w = w_0

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
        self.output = np.dot(self.w[:self.n], x) + self.w[-1]

        return self.activate(self.output) # Returns the output

    def calcpartialderivative(self, l1_deltas_x_w):
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
        dE_dO = sum(l1_deltas_x_w)
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

        # Return vector of delta * w
        # DON'T include bias
        return [delta * self.w[i] for i in range(self.n)]

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
            # self.n + 1 makes sure to include bias
            self.w[i] = self.w[i] - self.alpha * self.dE_dw[i]

class FullyConnectedLayer:
    def __init__(self, num_neurons, num_inputs, learning_rate=0.01, \
        activation='logistic', w_0=None):
        '''
        Initialize a fully connected layer in a neural network
        Arguments:
        ----------
        num_neurons: int
            - Number of neurons in layer
        num_inputs: int
            - Number of inputs to the layer (from previous layer)
        learning_rate: float, optional 
            - Default: 0.01
            - Learning rate to be used during gradient descent
        activation: string, optional
            - Default: 'logistic'
            - Options; 'logistic', 'linear'
            - Specifies the activation function to be used by each neuron in layer
        w_0: (num_neurons, num_inputs + 1) numpy array, optional
            - Default: None
            - If None, weights will be randomly initialized
            - If not None, must specify every weight, including bias

        Returns:
        --------
        No return value
        '''

        # Set numbers for layers:
        self.n_i = num_inputs
        self.n_n = num_neurons

        # Set up the neurons:
        self.neurons = []

        try:
            if w_0 == None:
                # Choose random values on uniform distribution in [0,1)
                self.w_0 = np.random.rand(num_neurons, num_inputs + 1)

        except ValueError: # Catches if w_0 is already
            self.w_0 = w_0

        # Creates an (n_n,) array of neurons
        for i in range(0, self.n_n):
            # Set up each neuron with desired properties
            new_neuron = Neuron(num_inputs = self.n_i,
                                activation = activation,
                                learning_rate = learning_rate,
                                w_0 = self.w_0[i,:])
            # Add this neuron to the list:
            self.neurons.append(new_neuron)

    def calculate(self, x):
        '''
        Calculates vector of outputs from neurons in layer
        Arguments:
        ----------
        x: array-like
            - Input to l (current) layer from l - 1 layer

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

    def calculatewdeltas(self, delta_w_matrix):
        ''' 
        Calculates list of deltas for each neuron in layer
            - Performs updates on the weights in the layer for each neuron
        Arguments:
        ----------
        w_delta: numpy array, size = (num_outputs, num_neurons)
            - Can also think of as: (# neurons in l + 1 layer, # neurons in l layer)
                - Current layer is l
            - Contains deltas * w's for layer l + 1 (each l + 1 neuron in rows)

        Returns:
        --------
        all_deltas: list
            - delta values for all neurons in layer
        ''' 

        # Initialize matrix
        # Matrix to be returned from FCL to the NeuralNetwork
        #   - Each row contains w*delta vector for each neuron in the layer
        new_delta_w = np.zeros(self.n_n, self.n_i)

        # Iterate over all neurons:
        for i in range(0, self.n_n):
            # Calculate delta value for this specific neuron:
            delta_i = self.neurons[i].calcpartialderivative(delta_w_matrix[:,i])
            # Add delta to the matrix
            new_delta_w[i,:] = delta_i
            # Update the weights in this neuron
            self.neurons[i].updateweights()
        
        return new_delta_w

class NeuralNetwork:
    def __init__(self, num_layers, num_neurons, num_inputs, lr=0.01, loss_func=None, \
        act_funcs=None):
        pass


if __name__ == '__main__':
    w_0 = np.array([[0, 0], [0, 0]])
    layer = FullyConnectedLayer(num_inputs = 2, num_neurons = 2, w_0 = w_0)
    print(layer.calculate([2, 3]))