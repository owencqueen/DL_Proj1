import sys
import numpy as np
import matplotlib.pyplot as plt
import copy

# COSC 525 Project 2: Owen Queen and Sai Thatigotla

def get_convolution_indices(tl_row, tl_col, depth, kernel_size):
    ''' 
    Given top-left row and top-left column indices, gets the indices needed
    for performing a convolution over that spot
    Arguments:
    ----------
    tl_row: int
        - Row index of the top-left spot in your convolution area
    tl_col: int
        - Column index of the top-left spot in your convolution area
    depth: int
        - Depth of the image/data you are convolving over
    kernel_size: int
        - Size of kernel used in convolution

    Returns:
    --------
    xy_matrices: list of tuples
        - Gives a list (1-dimensional) of tuples that correspond to indices to use in convolution
    '''
    xvals = np.array([[i] * kernel_size for i in range(tl_row, tl_row + kernel_size)]).flatten()
    yvals = list(range(tl_col, tl_col + kernel_size)) * 3

    xy = list(zip(xvals, yvals)) # Stacks xy's together
    xy_matrices = []
    for z in range(depth): 
        # Repeats xy list for every z value needed (corresponding to channels)
        for el in xy:
            xy_matrices.append((el[0], el[1], z))

    return xy_matrices

class Neuron:
    def __init__(self, num_inputs, w_0, activation = 'logistic',  \
                learning_rate = 0.01):
        '''
        Arguments:
        ----------
        num_inputs: int
            - Number of inputs to the neuron
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
        self.output_size = [num_neurons] # Need for NeuralNetwork compatibility

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

        return np.array(output)

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
        new_delta_w = np.zeros((self.n_n, self.n_i))

        # Iterate over all neurons:
        for i in range(0, self.n_n):
            # Calculate delta value for this specific neuron:
            delta_i = self.neurons[i].calcpartialderivative(delta_w_matrix[:,i])
            # Add delta to the matrix
            new_delta_w[i,:] = delta_i
            # Update the weights in this neuron
            self.neurons[i].updateweights()
        
        return new_delta_w

class ConvolutionalLayer:
    
    def __init__(self, kernel_num, kernel_size, input_size, lr = 0.01, \
                    activation = 'logistic', w_0 = None):
        ''' 
        Arguments:
        ----------
        kernel_num: int
            - Numbers of kernels in layer
        kernel_size: int
            - Size of kernels in layer
        input_size: (channels, x, y) array-like 
            - Size of input to layer
            - Channels MUST come as first in array-like
        lr: float, optional
            - Default: 0.01
            - Learning rate for all neurons in layer
        activation: string, optional
            - Default: 'logistic'
            - Activation function for the layer
            - Options: 'logistic', 'binary'
        w_0: np array
            - Must be of size (kernel_num, kernel_size, kernel_size)

        No return
        '''
        self.kernel_num = kernel_num
        self.kernel_size = kernel_size
        self.input_size = input_size[1:]
        self.input_channels = input_size[0]

        # Initialize neurons:
        self.output_size = [self.kernel_num, (input_size[1] - kernel_size) + 1, (input_size[2] - kernel_size) + 1]
        num_neurons = self.output_size[1] * self.output_size[2]

        #Random initialization of weights
        if w_0 is None:
            # Choose random values on uniform distribution in [0,1)
            # Size is <kernel number> x <self.input_channels> x <kernel_size> x <kernel_size>
            # Make each weight flat, with each index indicating a new weight
            self.w_0 = np.random.rand(self.kernel_num, self.input_channels *
                                self.kernel_size * self.kernel_size + 1)

        else: # Catches if w_0 is already given
            self.w_0 = w_0

        # Add each neuron:
        self.kernels = []
        for i in range(kernel_num):
            self.kernels.append([]) # Append empty 
            
            for j in range(num_neurons): # Build one kernel's neurons
                new_neuron = Neuron(num_inputs = \
                        self.kernel_size * self.kernel_size * self.input_channels,
                        activation = activation,
                        learning_rate = lr,
                        w_0 = self.w_0[i, :].flatten())
                # Must have shared weights across kernels (i.e. using i)
                # Each neuron has n*n*channels weights (input channels)
                self.kernels[-1].append(new_neuron) # Add new neuron
        # self.kernels[i] refers to the ith kernel's neurons

    def calculate(self, x):
        '''
        Calculate feedforward convolution for the layer
        Arguments:
        ----------
        x: np array
            - Must be of size self.input_size
            - Has three dimensions - (channels, x_input, y_input)
        
        Returns:
        --------
        output: np array
            - Complete output after all convolution operations on x
        '''
        
        # Output of feedforward convolution:
        output = np.zeros(self.output_size) 

        for k in range(self.kernel_num): # Over kernels
            for i in range(self.output_size[1]): # Over rows
                for j in range(self.output_size[2]): # Over cols
                    #top_left of input = (i, j)
                    
                    # Make the indices we need to iterate over
                    indices_to_get = get_convolution_indices(i, j, self.input_channels, self.kernel_size)

                    # Extract input to neuron from x
                    # Concatenate all together and flatten for input
                    # Get input from each channel, put into same neuron
                    neuron_input = np.array([x[channel][z][y] for z, y, channel in indices_to_get])
                        # Puts all input in a 1d array

                    # Save calculation of neuron to output matrix
                    # k - goes over kernels
                    # i - goes over rows of each input matrix
                    # j - goes over cols of each input matrix
                    output[k,i,j] = self.kernels[k][i + j].calculate(neuron_input)

        return output

    def calculatewdeltas(self, delta_w_matrix):
        '''
        This function has two main tasks:
            1. Calculate w*delta matrix to pass to l-1 layer
            2. Perform weight updates for each neuron (thereby each weight in kernels)

        Arguments:
        ----------
        delta_w_matrix: np array
            - Must be size of output
            - Calculated in l+1 layer
            - If coming from Flatten layer, input should be transformed

        Returns:
        --------
        dE_doutx: np array
            - Will be of size input_size
            - delta_w_matrix to be used in l-1 layer
        '''

        dE_doutx = np.zeros((self.input_channels, self.input_size[0], self.input_size[1]))

        # Performing convolutions to perform weight updates w/in each kernel
        for k in range(self.kernel_num): # Over kernels
            for i in range(self.output_size[1]): # Over rows
                for j in range(self.output_size[2]): # Over cols
                    # Get error for this neuron from the delta_w_ij matrix
                    #   Bulk of work is done in l+1 layer to calculate this value
                    delta_w_ij = [delta_w_matrix[k, i, j]] # Wrap in list for compatibility in neuron

                    # Calculate the partial derivatives
                    current_delta_w = self.kernels[k][i + j].calcpartialderivative(delta_w_ij)
                        # Should be size (self.input_channels x self.kernel_size x self.kernel_size), but in 1D
                    
                    # Therefore, we need to reshape it and perform the element-wise addition/convolution
                    current_delta_w = np.reshape(current_delta_w, (self.input_channels, self.kernel_size, self.kernel_size))

                    # We repeatedly add to the dE_doutx in an element-wise fashion
                    #   Do this over our neuron's context within the tensor
                    #   Note that the tensor is the same size as input
                    # Must do this over EVERY CHANNEL (i.e. : in first spot)
                    dE_doutx[:, i:(i + self.kernel_size), j:(j + self.kernel_size)] += current_delta_w
                    
                    # Update the weights for the neuron we're currently on
                    self.kernels[k][i + j].updateweights()

        return dE_doutx
class MaxPoolingLayer:
    """2D Max Pooling layer class
    """
    def __init__(self, kernel_size, input_dim):
        """Max pooling constructor

        Args:
            kernel_size (int): dimension of sqaure kernel
            input_dim (list): 3 element list (channels, x, y) of input shape to layer
        """
        self.k_s = kernel_size
        self.i_d = input_dim

        out = np.floor(((input_dim[1] - kernel_size)/kernel_size)+1)
        # size of output feature maps
        self.output_size = [input_dim[0], int(out), int(out)]

    def calculate(self, input):
        """Function for forward pass

        Args:
            input (ndarray): Numpy array of layer input (channels, x, y)

        Returns:
            ndarray: output array of max pooling
        """
        self.max_loc = np.zeros(input.shape)
        feature_map = np.zeros(self.output_size)

        for i in range(input.shape[0]):
            #iterate over channels
            for j in range(self.output_size[1]):
                for k in range(self.output_size[2]):
                    # Create sub array of input to pool over
                    sub_arr = input[i, (j*self.k_s): (j*self.k_s)+self.k_s, (k*self.k_s): (k*self.k_s)+self.k_s]
                    ind = np.unravel_index(np.argmax(sub_arr, axis=None), sub_arr.shape)
                    feature_map[i][j][k] = sub_arr[ind]
                    # Saves max location in input for backprop
                    self.max_loc[i][(j*self.k_s)+ind[0]][(k*self.k_s)+ind[1]] = 1

        return feature_map

    def calculatewdeltas(self, input):
        """Backpropagation for max pooling

        Args:
            input (ndarray): numpy array of input for backprop (channels, x ,y)

        Returns:
            ndarray: output of backpropogation of max pooling (channels, x, y)
        """
        output = copy.deepcopy(self.max_loc)

        for i in range(input.shape[0]):
            for j in range(input.shape[1]):
                for k in range(input.shape[2]):
                    # Multiply splice of output array by the update
                    output[i, (j*self.k_s): (j*self.k_s)+self.k_s, (k*self.k_s): (k*self.k_s)+self.k_s] *= input[i, j, k]

        return output

class FlattenLayer:
    """Flatten layer
    """
    def __init__(self, input_size):
        """Constructor

        Args:
            input_size (list): list of input size (channels, x, y)
        """
        self.i_s = input_size
        self.output_size = [self.i_s[0] * self.i_s[1] * self.i_s[2]]

    def calculate(self, input):
        """Reshapes input to flatten

        Args:
            input (ndarray): numpy array to flatten (channels, x, y)

        Returns:
            ndarray: flattened input 1D ndarray
        """
        return np.reshape(input, -1)
        
    def calculatewdeltas(self, input):
        """Reshape into original input

        Args:
            input (ndarray): 1D numpy array to reshape

        Returns:
            ndarray: (channels, x, y)
        """
        return np.reshape(input, self.i_s)

class NeuralNetwork:    #initialize with the number of layers, number of neurons in each layer (vector), input size, activation (for each layer), the loss function, the learning rate and a 3d matrix of weights weights (or else initialize randomly)    
    def __init__(self, inputSize, loss='square', lr=.001):
        '''
        Initializes the Neural Network
        Arguments:
        ----------
        inputSize: int
            - number of inputs
        loss: string, optional
            - Default: 'square'
            - Options: 'square' (square loss), 'binary' (binary cross-entropy loss)
            - loss function to use for network
        lr: float, optional
            - Default: 0.001
            - learning rate for backpropagation
        Returns
        -------
        No return value
        '''      
        self.in_size = inputSize
        self.lr = lr

        #set loss function
        if loss == 'binary':
            self.loss = lambda y_hat, y: np.sum([-(yt[0]*np.log(yh) + (1-yt[0])*np.log(1-yh)) for yh, yt in zip(y_hat, y)])/len(y)
            self.loss_deriv = lambda y_hat, y: -(y/y_hat) + ((1-y)/(1-y_hat))
        elif loss == 'square':
            self.loss = lambda y_hat, y: np.sum(np.square(y_hat - y))
            self.loss_deriv = lambda y_pred, y: 2 * (y_pred - y)

        #set up network
        self.network = []

    def addLayer(self, layer_type, num_neurons = 0, kernel_size = 3, num_kernels = 0, 
                    activation = 'logistic', weights = None):
        '''
        Adds a layer to the NeuralNetwork object
        Arguments:
        ----------
        layer_type: string
            - Denotes type of layer to be added
            - Options: 'Conv' (convolutional), 'FC' (fully-connected), 'Pool' (Max Pooling), and 'Flatten' (flattening layer)
        num_neurons: int, optional
            - Default:
            - Required in: 'FC'
                - If given 'FC' as type, will throw an error 
        kernel_size: int, optional
            - Required in: 'Conv', 'Pool'
        num_kernels: int, optional
            - Required in: 'Conv'
        activation: string, optional
            - Default: 'logistic'
            - Activation function to use throughout the additional layer
            - If layer_type == 'Flatten' or 'Pool', this is ignored
        weights: np array, optional
            - Default: None
            - If None, weights are generated randomly
            - Weights must match dimensions specified by your given layer
            - If layer_type == 'Flatten' or 'Pool', this is ignored

        No return value
        ''' 

        # Get input size from previous layer
        if len(self.network) > 0:
            input_size = self.network[-1].output_size
        #elif input_size is None:
        else:
            # If no layers added, get input size to entire network
            input_size = self.in_size

        if layer_type == 'Conv':
            new_layer = ConvolutionalLayer(num_kernels, kernel_size, input_size, self.lr, activation, w_0 = weights)
            self.network.append(new_layer)
            
        elif layer_type == 'FC':

            if (len(input_size) > 1):
                # Throw error if user forgets a flatten layer
                print('Must have Flatten Layer before FC')
                exit

            # Note: if weights are left to be generated randomly, will be done in layer
            new_layer = FullyConnectedLayer(num_neurons, input_size[0], self.lr, activation, w_0 = weights)
            self.network.append(new_layer)

        elif layer_type == 'Pool':
            # blank until pool and flatten added
            new_layer = MaxPoolingLayer(kernel_size = kernel_size, input_dim = input_size)
            self.network.append(new_layer)

        elif layer_type == 'Flatten':
            # blank until pool and flatten added
            new_layer = FlattenLayer(input_size)
            self.network.append(new_layer)

    
    #Given an input, calculate the output (using the layers calculate() method)    
    def calculate(self,input):
        '''
        Forward pass of network
        Arguments:
        -----------
        input: numpy array
            - input to network

        Returns:
        --------
        out: numpy array
            - output from network's ouptut layer
        '''          
        out = input
        # Number of hidden layers after first layer
        for i in range(0, len(self.network)):
            out = self.network[i].calculate(out)

        return out

    #Given a predicted output and ground truth output simply return the loss (depending on the loss function)    
    def calculateloss(self,yp,y):
        '''
        Calculates loss value
        Arguments:
        ----------
        yp: numpy array
            - prediction from network
        y: numpy array
            - ground truth

        Returns:
        --------
        Returns value of loss function
        '''     
        return self.loss(yp, y)    
        
    #Given a predicted output and ground truth output simply return the derivative of the loss (depending on the loss function)            
    def lossderiv(self,yp,y):
        '''
        Calculates derivative of loss function
        Arguments:
        ----------
        yp: float
            - prediction from one of output neurons
        y: float
            - ground truth that corresponds to yp

        Returns:
        --------
        Returns value of loss derivative
        '''        
        return self.loss_deriv(yp, y)    
        
    #Given a single input and desired output perform one step of backpropagation (including a forward pass, getting the derivative of the loss, and then calling calcwdeltas for layers with the right values             
    def train(self,x,y):
        '''
        Backpropogation function
        Arguments:
        ----------
        x: numpy array
            - input for network to train on
        y: numpy array
            - ground truth for network

        Returns:
        --------
        No return value
        '''        
        pred = self.calculate(x)

        # Get number of neurons in last layer:
        n_last = self.network[-1].n_n

        # Set up delta
        delt = np.zeros((1, n_last))

        # Calculate each loss function for start of backprop in last layer
        for i in range(0, n_last):
            delt[:, i] = self.lossderiv(np.array(pred)[i], y[i])

        # Flow of delta*w's backwards through network
        for j in range(len(self.network) - 1, -1, -1):
            delt = self.network[j].calculatewdeltas(delt)