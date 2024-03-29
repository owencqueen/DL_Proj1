import sys
import numpy as np
import matplotlib.pyplot as plt

# COSC 525 Project 1: Owen Queen and Sai Thatigotla

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

class NeuralNetwork:    #initialize with the number of layers, number of neurons in each layer (vector), input size, activation (for each layer), the loss function, the learning rate and a 3d matrix of weights weights (or else initialize randomly)    
    def __init__(self,numOfLayers,numOfNeurons, inputSize, activation='logistic', loss='square', lr=.001, weights=None):
        '''
        Initializes the Neural Network
        Arguments:
        ----------
        numOfLayers: int
            - number of hidden + output layers
        numOfNeurons: (numOfLayers, ) list
            - number of neurons in each layer
            - numOfNeurons[i] should be the number of neurons in the ith layer
        inputSize: int
            - number of inputs
        activation: string, optional
            - Default: 'logistic'
            - Options: 'logistic', 'linear'
            - Specifies the activation function to be used by each neuron in layer
        loss: string, optional
            - Default: 'square'
            - Options: 'square' (square loss), 'binary' (binary cross-entropy loss)
            - loss function to use for network
        lr: float, optional
            - Default: 0.001
            - learning rate for backpropagation
        weights: numpy array, optional
            - Default: None
            - weights to load into network
        Returns
        -------
        No return value
        '''      

        self.n_l = numOfLayers
        self.n_n = numOfNeurons
        self.in_size = inputSize

        #set loss function
        if loss == 'binary':
            self.loss = lambda y_hat, y: np.sum([-(yt[0]*np.log(yh) + (1-yt[0])*np.log(1-yh)) for yh, yt in zip(y_hat, y)])/len(y)
            self.loss_deriv = lambda y_hat, y: -(y/y_hat) + ((1-y)/(1-y_hat))
        elif loss == 'square':
            self.loss = lambda y_hat, y: 0.5 * np.sum(np.square(y_hat - y))
            self.loss_deriv = lambda y_pred, y: -(y-y_pred)

        #set up network
        self.network = []

        #set up input layer
        in_layer = []
        if weights is None:
            in_layer = FullyConnectedLayer(self.n_n[0], self.in_size, activation=activation, learning_rate=lr)
        else:
            in_layer = FullyConnectedLayer(self.n_n[0], self.in_size, activation=activation, learning_rate=lr, w_0=weights[0])
        self.network.append(in_layer)

        # Set every layer thereafter
        tmp_layer = []
        for i in range(1, self.n_l):
            if weights is None:
                tmp_layer = FullyConnectedLayer(self.n_n[i], self.n_n[i-1], activation=activation, learning_rate=lr)
            else:
                tmp_layer = FullyConnectedLayer(self.n_n[i], self.n_n[i-1], activation=activation, learning_rate=lr, w_0=weights[i])
            self.network.append(tmp_layer)
    
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
        for i in range(1, self.n_l):
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

        delt = np.zeros((1, self.n_n[-1]))

        # Calculate each loss function for start of backprop in last layer
        for i in range(0, self.n_n[-1]):
            delt[:, i] = self.lossderiv(np.array(pred)[i], y[i])

        # Flow of delta*w's backwards through network
        for j in range(self.n_l-1, -1, -1):
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