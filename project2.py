import sys
import numpy as np
import matplotlib.pyplot as plt
import copy

# COSC 525 Project 2: Owen Queen and Sai Thatigotla

def get_convolution_indices(tl_row, tl_col, depth, kernel_size):
    ''' 
    Given top-left row and top-left column indices, gets the indices needed
    for performing a convolution over that spot
    '''
    xvals = np.array([[i] * kernel_size for i in range(tl_row, tl_row + kernel_size)]).flatten()
    yvals = list(range(tl_col, tl_col + kernel_size)) * 3

    #indices_to_get = list(zip(xvals, yvals))
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
        input_size: (channels, x, y) array-like 
        
        weights: (num_kernels, kernel_size, kernel_size)

        '''
        self.kernel_num = kernel_num
        self.kernel_size = kernel_size
        self.input_size = input_size[1:]
        self.input_channels = input_size[0]

        # Initialize neurons:
        self.output_size = [(input_size[0] - kernel_size) + 1, (input_size[1] - kernel_size) + 1, self.kernel_num]
        num_neurons = self.output_size[0] * self.output_size[1]

        #Random initialization of weights
        try:
            if w_0 == None:
                # Choose random values on uniform distribution in [0,1)
                # Size is <kernel number> x <self.input_channels> x <kernel_size> x <kernel_size>
                # Make each weight flat, with each index indicating a new weight
                self.w_0 = np.random.rand(self.kernel_num, self.input_channels *
                                    self.kernel_size * self.kernel_size + 1)

        except ValueError: # Catches if w_0 is already given
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
                # Weights for each neuron: w1, w2, ..., w9, w1, w2, ..., w9
                # Must have shared weights across kernels (i.e. using i)
                # Each neuron has n*n*channels weights (input channels)
                self.kernels[-1].append(new_neuron) # Add new neuron

        # self.kernels[i] refers to the ith kernel's neurons

    def calculate(self, x):
        '''x has three dims - (channels, x_input, y_input)'''

        #np[channel]

        #.reshape()
        
        # Output of feedforward convolution:
        output = np.zeros((self.kernel_num, self.output_size[0], self.output_size[1])) 

        for k in range(self.kernel_num): # Over kernels
            for i in range(self.output_size[0]): # Over rows
                for j in range(self.output_size[1]): # Over cols
                    
                    #top_left of input = (i, j)
                    
                    # Make the indices we need to iterate over
                    indices_to_get = get_convolution_indices(i, j, self.input_channels, self.kernel_size)

                    # Extract input to neuron from x
                    # Concatenate all together and flatten for input
                    # Get input from each channel, put into same neuron
                    neuron_input = np.array([x[channel][x][y] for x, y, channel in indices_to_get])
                        # Puts all input in a 1d array

                    #for channel in range(self.input_channels):
                        
                        #neuron_input += [x[channel][ind[0]][ind[1]] for ind in indices_to_get]

                    # Save calculation of neuron to output matrix
                    # k - goes over kernels
                    # i - goes over rows of each input matrix
                    # j - goes over cols of each input matrix
                    output[k,i,j] = self.kernel[k][i + j].calculate(neuron_input)

        return output

    def calculatewdeltas(self, delta_w_matrix):
        '''
        Two main tasks:
            1. Calculate w*delta matrix to pass to l-1 layer
            2. Perform weight updates for each neuron (thereby each weight in kernels)

        Arguments:
        ----------
        delta_w_matrix: (output_size[0], output_size[1]) array
            - Must be of this dimension for compatibility
            - Only one channel - acts as if its repeated over multiple channels
        '''

        # Reshaping delta_w if needed:
        # IRRELEVANT -------------------
        #delta_w_matrix = delta_w_matrix.reshape((delta_w_matrix.shape[0], 1, delta_w_matrix.shape[1]))

        next_dw_mat = []

        dE_doutx = np.zeros((self.input_channels, self.input_size[0], self.input_size[1]))

        # Performing convolutions to perform weight updates w/in each kernel
        for k in range(self.kernel_num): # Over kernels
            # Setting up matrix of zeros for dE's wrt each w in kernel - calculated in Neuron class
            #dE_dwi_matrix = np.zeros((self.kernel_size, self.kernel_size))

            # dE_doutx should be same size as input to layer
            # [0] is height, [1] is width
            #dE_doutx = np.zeros((self.input_size[0], self.input_size[1]))
            

            for i in range(self.output_size[0]): # Over rows
                for j in range(self.output_size[1]): # Over cols
                    # Make the indices we need to consider in dE_doutx
                    # These indices correspond to our current neuron
                    conv_inds = get_convolution_indices(i, j, self.input_channels, self.kernel_size)
                        # i: top-left x value
                        # j: top-left y values
                        # self.input_channels: depth/num. channels in input
                        # self.kernel_size: size of kernel

                    # Get neuron's context (for a given kernel k) from delta_w matrix
                    delta_w_ij = [delta_w_matrix[channel, x, y] for x, y, channel in conv_inds]

                    # Calculate the partial derivatives
                    current_delta_w = self.kernel[k][i + j].calcpartialderivative(delta_w_ij)
                        # Should be size (self.input_channels x self.kernel_size x self.kernel_size), but in 1D
                    
                    # Therefore, we need to reshape it and perform the element-wise addition/convolution
                    current_delta_w = np.reshape(current_delta_w, (self.input_channels, self.kernel_size, self.kernel_size))

                    # We repeatedly add to the dE_doutx in an element-wise fashion
                    #   Do this over our neuron's context within the tensor
                    #   Note that the tensor is the same size as input
                    # Must do this over EVERY CHANNEL (i.e. : in first spot)
                    dE_doutx[:, i:(i + self.kernel_size), j:(j + self.kernel_size)] += current_delta_w

                    # Need to pass convolved version of delta_w matrix backwards:

                    # Also pass delta_w from l+1 portion backwards
                    #current_delta_w = self.kernel[k][i + j].calcpartialderivative(delta_w_matrix[k, :,(i + j)])
                    
                    # Update the weights for the neuron we're currently on
                    self.kernel[k][i + j].updateweights()

                    # Add current_delta_w to appropriate location (conv_inds) in dE_doutx
                    #for i in len(current_delta_w):
                    #    ci, cj = conv_inds[i]
                        #dE_doutx[ci, cj] += current_delta_w
                        #dE_doutx[ci + cj] += current_delta_w

            #next_dw_mat.append(dE_doutx) # Append to list that will comprise dw matrix

        #next_dw_mat = np.array(next_dw_mat)
        dE_doutx.shape == input_size

        return dE_doutx

#TODO : Add channels for arrays
class MaxPoolingLayer:
    def __init__(self, kernel_size, input_dim):
        self.k_s = kernel_size
        self.i_d = input_dim

        out = ((input_dim[1] - kernel_size)/kernel_size)+1
        self.output_size = [input_dim[0], out, out]

    def calculate(self, input):
        self.max_loc = np.zeros(input.shape)
        feature_map = np.array(output_size)

        for i in range(input.shape[0]):
            #iterate over channels
            for j in range(output_size[1]):
                for k in range(output_size[2]):
                    sub_arr = input[i, (j*k_s): (j*k_s)+k_s, (k*k_s): (k*k_s)+k_s]
                    ind = np.unravel_index(np.argmax(sub_arr, axis=None), sub_arr.shape)
                    feature_map[i][j][k] = sub_arr[ind]
                    max_loc[i][(j*k_s)+ind[0]][(k*k_s)+ind[1]] = 1

        """
        out_dim = ((i_d - k_s)/k_s)+1
        feature_map = np.array((out_dim, out_dim))
        for i in range(out_dim):
            for j in range(out_dim):
                sub_arr = input[(i*k_s): (i*k_s)+k_s, (j*k_s): (j*k_s)+k_s]
                ind = np.unravel_index(np.argmax(sub_arr, axis=None), sub_arr.shape)
                feature_map[i][j] = sub_arr[ind]
                max_loc[(i*k_s)+ind[0]][(j*k_s)+ind[1]] = 1
        """

        #save indexes better?

        return feature_map

    def calculatewdeltas(self, input):
        output = copy.deepcopy(max_loc)

        for i in range(input.shape[0]):
            for j in range(input.shape[1]):
                for k in range(input.shape[2]):
                    output[i, (j*k_s): (j*k_s)+k_s, (k*k_s): (k*k_s)+k_s] *= input[i, j, k]

        return output

class FlattenLayer:
    def __init__(self, input):
        self.i_s = input.shape
        self.output_size = [i_s[0]*i_s[1]*i_s[2], 1]

    def calculate(self, input):
        return np.reshape(input, -1)
        
    def calculatewdeltas(self, input):
        return np.reshape(input, i_s)

class NeuralNetwork:    #initialize with the number of layers, number of neurons in each layer (vector), input size, activation (for each layer), the loss function, the learning rate and a 3d matrix of weights weights (or else initialize randomly)    
    #def __init__(self,numOfLayers,numOfNeurons, inputSize, activation='logistic', loss='square', lr=.001, weights=None):
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
            self.loss = lambda y_hat, y: 0.5 * np.sum(np.square(y_hat - y))
            self.loss_deriv = lambda y_pred, y: -(y-y_pred)

        #set up network
        self.network = []

    def addLayer(self, layer_type, num_neurons = 0, kernel_size = 3, num_kernels = 0, 
                    activation = 'sigmoid', weights = None):

        '''
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
        ''' 

        # Get input size from previous layer
        if len(self.network) > 0:
            input_size = self.network[-1].output_size
        elif input_size is None:
            # If no layers added, get input size to entire network
            input_size = self.in_size

        if layer_type == 'Conv':
            new_layer = ConvolutionalLayer(num_kernels, kernel_size, input_size, self.lr, activation, 'w_0')
            self.network.append(new_layer)
            
        elif layer_type == 'FC':

            if (len(input_size) > 1):
                # Throw error if user forgets a flatten layer
                print('Must have Flatten Layer before FC')
                exit

            # Note: if weights are left to be generated randomly, will be done in layer
            new_layer = FullyConnectedLayer(num_neurons, input_size, self.lr, activation, w_0 = weights)
            self.network.append(new_layer)

        elif layer_type == 'Pool':
            pass # blank until pool and flatten added

        elif layer_type == 'Flatten':
            pass # blank until pool and flatten added
    
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
        #calculate first layer output based on input           
        out = self.network[0].calculate(input)
        # #number of hidden layers after first layer
        for i in range(0, range(len(self.network))):
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

def check_logical_predictions(y_hat, y):
    '''
    Checks if predictions made for logical problems are correct (and, xor)
    Arguments:
    ----------
    y_hat: (n, ) numpy array
        - Predicted
    y: (n, ) numpy array
        - Ground truth

    Returns:
    --------
    result: bool
        - True if y_hat == y
        - False if y_hat != y
    '''
    for i in range(y_hat.shape[0]):
        if (y[i] != y_hat[i]):
            return False
    return True

def plot_one_loss_curve(losses, ep = None, title = 'Loss per Epochs'):
    '''
    Plots a singular loss curve over epochs
    Arguments:
    ----------
    losses: (n, ) list
        - Loss for network on each epoch
        - n is number of epochs
    ep: int, optional
        - Default: None
        - If None, doesn't plot the vertical line
        - Epoch at which the network converged
        - Will draw a vertical line at this point
    title: string, optional
        - Default: 'Loss per Epochs'
        - Title for plot

    No return value
    '''
    plt.plot(losses, c = 'g')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title(title)
    if ep is not None:
        plt.vlines(ep, ymax = max(losses), ymin = min(losses), label = 'Epoch of Correct Predictions')
        plt.legend()

    plt.grid()
    plt.show()

if __name__ == '__main__':

    if len(sys.argv) < 2:
        # Error checking for correct input
        print('usage: python3 project1.py <option>')
        print('\t Options: example, and, xor')
        exit

    elif sys.argv[1] == 'example':
        w = np.array([[[.15,.2,.35],[.25,.3,.35]],[[.4,.45,.6],[.5,.55,.6]]])        
        x = np.array([0.05,0.1])        
        y = np.array([0.01,0.99])

        # Neural network with 2 layers, 2 neurons per layer
        nn = NeuralNetwork(2, [2, 2], 2, weights=w, lr=0.5, loss='square')
        net_loss = []
        nn.train(x, y)
        print('Calculated Outputs (after 1 epoch) =', nn.calculate(x))
        print("Weights in Network (please refer to in-class example for each weight's label):")

        print('Weight \t Value')
        # Iterate over neuron h1, h2:
        count = 1
        bcount = 1
        for i in [0, 1]:
            for j in [0, 1]:
                print('w{} \t {}'.format(count, nn.network[i].neurons[j].w[0]))
                print('w{} \t {}'.format(count + 1, nn.network[i].neurons[j].w[1]))
                print('b{} \t {}'.format(bcount, nn.network[i].neurons[j].w[2]))
                bcount += 1
                count += 2

        print('Loss (MSE) after 1 Epoch =', nn.calculateloss(nn.calculate(x), y))


    elif sys.argv[1] == 'and':
        # Initializing with random weights
        x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [0], [0], [1]]) # Must wrap each label in a list (for generality)
        nn = NeuralNetwork(1, [1], 2, lr = 0.1, loss = 'binary')

        net_loss = []
        first = True
        for i in range(0, 100): #100 is maximum
            for j in range(0, len(x)):
                nn.train(x[j], y[j])

            y_hat = [nn.calculate(xi) for xi in x]
            #for j in range(0, len(x)):
            #    y_hat.append(nn.calculate(x[j]))
                
            net_loss.append(nn.calculateloss(np.array(y_hat), y))

            # Classify the predictions based on definition of sigmoid
            y_hat_preds = np.array([0 if yh < 0.5 else 1 for yh in y_hat])

            # Stop epochs early if predictions are correct:
            if (check_logical_predictions(y_hat_preds, [0, 0, 0, 1]) and first):
                first = False
                ep = i

        # Printing the final predictions:
        print('Running AND Logic Data (1 Perceptron Network)')
        print('Input \t Prediction \t Ground Truth')
        for i in range(len(y_hat)):
            print('{} \t {:.6f} \t {}'.format(x[i], y_hat[i][0], y[i][0]))
        print('Epoch of Convergence', ep)
        print('Final Loss (Binary Cross Entropy) =', net_loss[-1])

        # Plot the loss curve
        plot_one_loss_curve(net_loss, ep = ep, title = 'Loss per Epoch (AND)')

    elif sys.argv[1] == 'xor':
        x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])

        # Neural net with 1 perceptron
        nn = NeuralNetwork(1, [1], 2, lr = 0.5, loss = 'binary')

        net_loss = []

        # Run for 100 epochs
        for i in range(0, 10000):
            for j in range(0, len(x)):
                nn.train(x[j], y[j])

            y_hat = [nn.calculate(xi) for xi in x]
                
            net_loss.append(nn.calculateloss(np.array(y_hat), y))

            # Classify the predictions based on definition of sigmoid
            y_hat_preds = np.array([0 if yh < 0.5 else 1 for yh in y_hat])

        # Printing the final predictions:
        print('Running XOR Logic Data (Single Perceptron)')
        print('Input \t Prediction \t Ground Truth')
        for i in range(len(y_hat)):
            print('{} \t {:.6f} \t {}'.format(x[i], y_hat[i][0], y[i][0]))
        print('Final Loss (Binary Cross Entropy) =', net_loss[-1])
        print('')

        plot_one_loss_curve(net_loss, title = 'Single Perceptron Loss vs. Epoch (XOR)')

        # ---------------------------
        # Neural net with 2 layers
        nn = NeuralNetwork(2, [2, 1], 2, lr = 1, loss = 'binary')

        net_loss = []
        first = True

        # Run for 100 epochs
        for i in range(0, 10000):
            for j in range(0, len(x)):
                nn.train(x[j], y[j])

            y_hat = [nn.calculate(xi) for xi in x]
                
            net_loss.append(nn.calculateloss(np.array(y_hat), y))

            # Classify the predictions based on definition of sigmoid
            y_hat_preds = np.array([0 if yh < 0.5 else 1 for yh in y_hat])

            # Stop epochs early if predictions are correct:
            if (check_logical_predictions(y_hat_preds, [0, 1, 1, 0]) and first):
                first = False
                ep = i

        # Printing the final predictions:
        print('Running XOR Logic Data (Network with 1 Hidden Layer)')
        print('Input \t Prediction \t Ground Truth')
        for i in range(len(y_hat)):
            print('{} \t {:.6f} \t {}'.format(x[i], y_hat[i][0], y[i][0]))
        print('Epoch of Convergence', ep)
        print('Final Loss (Binary Cross Entropy) =', net_loss[-1])
        print('')

        # Plot the loss curve
        plot_one_loss_curve(net_loss, ep = ep, title = '1 Hidden Layer Loss vs. Epoch (XOR)')
