import sys
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers

from project2 import *

def generateExample1():
    # Gets consistent weights in same manner as 2, 3 versions
    np.random.seed(20)

    net_input = np.random.rand(5, 5)
    net_output = np.random.rand(1)

    l1k1 = np.random.rand(3, 3, 1, 1)
    l1b1 = np.random.rand(1)

    l2w = np.random.rand(9, 1)
    l2b = np.random.rand(1)

    return l1k1, l1b1, l2w, l2b, net_input, net_output

# COPIED FROM EXAMPLE CODE --------------------------------------
#Generate data and weights for "example2"
def generateExample2():
    #Set a seed (that way you get the same values when rerunning the function)
    np.random.seed(10)

    #First hidden layer, two kernels
    l1k1=np.random.rand(3,3)
    l1k2=np.random.rand(3,3)
    l1b1=np.random.rand(1)
    l1b2=np.random.rand(1)

    #second hidden layer, one kernel, two channels
    l2c1=np.random.rand(3,3)
    l2c2=np.random.rand(3,3)
    l2b=np.random.rand(1)

    #output layer, fully connected
    l3=np.random.rand(1,9)
    l3b=np.random.rand(1)

    #input and output
    input=np.random.rand(7,7)
    output=np.random.rand(1)

    return l1k1,l1k2,l1b1,l1b2,l2c1,l2c2,l2b,l3,l3b,input,output

def generateExample3():
    np.random.seed(30)
    
    # input + output for network
    input = np.random.rand(8, 8)
    output = np.random.rand(1)

    # Convolutional layer kernels and biases
    l1k1 = np.random.rand(3, 3, 1, 1)
    l1b1 = np.random.rand(1)
    l1k2 = np.random.rand(3, 3, 1, 1)
    l1b2 = np.random.rand(1)

    # Final layer weights and bias
    l3w = np.random.rand(18, 1)
    l3b = np.random.rand(1)

    return l1k1, l1b1, l1k2, l1b2, l3w, l3b, input, output

def run_example1(option = 'keras'):
    '''
    Runs example 1
    Arguments:
    ----------
    option: string, optional
        - Default: 'keras'
        - Options: 'keras' or 'project'
    '''

    l1k1, l1b1, l2w, l2b, input, output = generateExample1()

    if option == 'keras':
        model = Sequential()
        model.add(layers.Conv2D(1, 3, input_shape = (5, 5, 1), activation = 'sigmoid'))
        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation = 'sigmoid'))

        model.layers[0].set_weights([l1k1, l1b1])
        model.layers[2].set_weights([l2w, l2b])

        # Expand dimensions of input image
        img=np.expand_dims(input,axis=(0,3))

        # Printing parameters:
        np.set_printoptions(precision=5)
        print('model output before:')
        print(model.predict(img))
        sgd = optimizers.SGD(lr=100)
        model.compile(loss='MSE', optimizer=sgd, metrics=['accuracy'])
        history=model.fit(img,output,batch_size=1,epochs=1)
        print('model output after:')
        print(model.predict(img))

        print('1st convolutional layer, 1st kernel weights:')
        print(np.squeeze(model.get_weights()[0][:,:,0,0]))
        print('1st convolutional layer, 1st kernel bias:')
        print(np.squeeze(model.get_weights()[1][0]))

        print('fully connected layer weights:')
        print(np.squeeze(model.get_weights()[2]))
        print('fully connected layer bias:')
        print(np.squeeze(model.get_weights()[3]))
    
    elif option == 'project':

        # Add layers + initialize with weights
        cnn = NeuralNetwork(inputSize = (1, 5, 5), loss = 'square', lr = 100)
        cnn.addLayer(layer_type = 'Conv', kernel_size = 3, num_kernels = 1, \
            activation = 'logistic', weights = np.concatenate((l1k1.flatten(), l1b1)).reshape(1,10))
        cnn.addLayer(layer_type = 'Flatten')
        cnn.addLayer(layer_type = 'FC', num_neurons = 1, activation = 'logistic', \
            weights = np.concatenate((l2w.flatten(), l2b)).reshape(1,np.concatenate((l2w.flatten(), l2b)).shape[0]))

        np.set_printoptions(precision = 5)

        input = np.reshape(input, (1, 5, 5))

        print('model output before:')
        print(cnn.calculate(input))

        # Train the network
        cnn.train(input, output)
        out = cnn.calculate(input)
        print(out)

        # Get weights:
        kernel1 = cnn.network[0].kernels[0][0].w
        print('1st convolutional layer, 1st kernel weights:')
        print(kernel1[:-1].reshape((3, 3)))
        print('1st convolutional layer, 1st kernel bias:')
        print(kernel1[-1])

        fc = cnn.network[-1].neurons[0].w
        print('fully connected layer weights:')
        print(fc[:-1])
        print('fully connected layer bias:')
        print(fc[-1])


def run_example2(option = 'keras'):
    '''
    Runs example 2
        - Keras code adapted from Dr. Sadovnik's posted code
    Arguments:
    ----------
    option: string, optional
        - Default: 'keras'
        - Options: 'keras' or 'project'
    '''

    # Call weight/data generating function
    l1k1,l1k2,l1b1,l1b2,l2c1,l2c2,l2b,l3,l3b,input, output = generateExample2()

    if option == 'keras':
        #Create a feed forward network
        model=Sequential()

        # Add convolutional layers, flatten, and fully connected layer
        model.add(layers.Conv2D(2,3,input_shape=(7,7,1),activation='sigmoid'))
        model.add(layers.Conv2D(1,3,activation='sigmoid'))
        model.add(layers.Flatten())
        model.add(layers.Dense(1,activation='sigmoid'))

        #Set weights to desired values 

        #setting weights and bias of first layer.
        l1k1=l1k1.reshape(3,3,1,1)
        l1k2=l1k2.reshape(3,3,1,1)

        w1=np.concatenate((l1k1,l1k2),axis=3)
        model.layers[0].set_weights([w1,np.array([l1b1[0],l1b2[0]])]) #Shape of weight matrix is (w,h,input_channels,kernels)

        #setting weights and bias of second layer.
        l2c1=l2c1.reshape(3,3,1,1)
        l2c2=l2c2.reshape(3,3,1,1)

        w1=np.concatenate((l2c1,l2c2),axis=2)
        model.layers[1].set_weights([w1,l2b])

        #setting weights and bias of fully connected layer.
        model.layers[3].set_weights([np.transpose(l3),l3b])

        #Setting input. Tensor flow is expecting a 4d array since the first dimension is the batch size (here we set it to one), and third dimension is channels
        img=np.expand_dims(input,axis=(0,3))

        #print needed values.
        np.set_printoptions(precision=5)
        print('model output before:')
        print(model.predict(img))
        sgd = optimizers.SGD(lr=100)
        model.compile(loss='MSE', optimizer=sgd, metrics=['accuracy'])
        history=model.fit(img,output,batch_size=1,epochs=1)
        print('model output after:')
        print(model.predict(img))

        print('1st convolutional layer, 1st kernel weights:')
        print(np.squeeze(model.get_weights()[0][:,:,0,0]))
        print('1st convolutional layer, 1st kernel bias:')
        print(np.squeeze(model.get_weights()[1][0]))

        print('1st convolutional layer, 2nd kernel weights:')
        print(np.squeeze(model.get_weights()[0][:,:,0,1]))
        print('1st convolutional layer, 2nd kernel bias:')
        print(np.squeeze(model.get_weights()[1][1]))


        print('2nd convolutional layer weights:')
        print(np.squeeze(model.get_weights()[2][:,:,0,0]))
        print(np.squeeze(model.get_weights()[2][:,:,1,0]))
        print('2nd convolutional layer bias:')
        print(np.squeeze(model.get_weights()[3]))

        print('fully connected layer weights:')
        print(np.squeeze(model.get_weights()[4]))
        print('fully connected layer bias:')
        print(np.squeeze(model.get_weights()[5]))
    
    elif option == 'project':
        cnn = NeuralNetwork(inputSize = (1, 7, 7), loss = 'square', lr = 100)
        cnn.addLayer(layer_type = 'Conv', kernel_size = 3, num_kernels = 2, \
            activation = 'logistic', weights = np.append(
                np.concatenate((l1k1.flatten(), l1b1)).reshape(1,np.concatenate((l1k1.flatten(), l1b1)).shape[0]),
                np.concatenate((l1k2.flatten(), l1b2)).reshape(1,np.concatenate((l1k2.flatten(), l1b1)).shape[0]),
                axis=0))

        new_wts = np.concatenate((l2c1, l2c2, l2b), axis=None).reshape(1, np.concatenate((l2c1, l2c2, l2b), axis=None).shape[0])
        cnn.addLayer(layer_type='Conv', kernel_size=3, num_kernels=1, \
            activation='logistic', weights=new_wts)
        cnn.addLayer(layer_type = 'Flatten')
        cnn.addLayer(layer_type = 'FC', num_neurons = 1, activation = 'logistic', \
            weights = np.concatenate((l3.flatten(), l3b)).reshape(1,np.concatenate((l3.flatten(), l3b)).shape[0]))

        np.set_printoptions(precision = 5)

        input = np.reshape(input, (1, 7, 7))

        print('model output before:')
        print(cnn.calculate(input))

        # Train the network
        cnn.train(input, output)
        out = cnn.calculate(input)
        print('model output after:')
        print(out)

        np.set_printoptions(precision=5)
        # Get weights:
        kernel1 = cnn.network[0].kernels[0][0].w
        print('1st convolutional layer, 1st kernel weights:')
        print(kernel1[:-1].reshape((3, 3)))
        print('1st convolutional layer, 1st kernel bias:')
        print(kernel1[-1])

        kernel2 = cnn.network[0].kernels[1][0].w
        print('1st convolutional layer, 2nd kernel weights:')
        print(kernel2[:-1].reshape((3, 3)))
        print('1st convolutional layer, 2nd kernel bias:')
        print(kernel2[-1])

        kernel1_layer2 = cnn.network[1].kernels[0][0].w
        print('2nd convolutional layer weights:')
        print(kernel1_layer2[:-1].reshape(2, 3, 3))
        print('2nd convolutional layer bias:')
        print(kernel1_layer2[-1])

        fc = cnn.network[-1].neurons[0].w
        print('fully connected layer weights:')
        print(fc[:-1])
        print('fully connected layer bias:')
        print(fc[-1])

def run_example3(option = 'keras'):
    '''
    Runs example 3
    Arguments:
    ----------
    option: string, optional
        - Default: 'keras'
        - Options: 'keras' or 'project'
    '''

    l1k1, l1b1, l1k2, l1b2, l3w, l3b, input, output = generateExample3()

    if option == 'keras':
        model = Sequential()
        model.add(layers.Conv2D(2, 3, input_shape = (8, 8, 1), activation = 'sigmoid'))
        model.add(layers.MaxPool2D((2, 2), strides = 2))
        model.add(layers.Flatten())
        model.add(layers.Dense(1, activation = 'sigmoid'))

        # Initialize the weights
        model.layers[0].set_weights([np.concatenate((l1k1, l1k2), axis = 3), np.array([l1b1[0], l1b2[0]])])
        model.layers[3].set_weights([l3w, l3b])

        # Expand dimensions of input image
        img=np.expand_dims(input,axis=(0,3))

        # Printing parameters:
        np.set_printoptions(precision=5)
        print('model output before:')
        print(model.predict(img))
        sgd = optimizers.SGD(lr=100)
        model.compile(loss='MSE', optimizer=sgd, metrics=['accuracy'])
        history=model.fit(img,output,batch_size=1,epochs=1)
        print('model output after:')
        print(model.predict(img))

        print('1st convolutional layer, 1st kernel weights:')
        print(np.squeeze(model.get_weights()[0][:,:,0,0]))
        print('1st convolutional layer, 1st kernel bias:')
        print(np.squeeze(model.get_weights()[1][0]))

        print('1st convolutional layer, 2nd kernel weights:')
        print(np.squeeze(model.get_weights()[0][:,:,0,1]))
        print('1st convolutional layer, 2nd kernel bias:')
        print(np.squeeze(model.get_weights()[1][1]))

        print('fully connected layer weights:')
        print(np.squeeze(model.get_weights()[2]))
        print('fully connected layer bias:')
        print(np.squeeze(model.get_weights()[3]))

    elif option == 'project':
        cnn = NeuralNetwork(inputSize = (1, 8, 8), loss = 'square', lr = 100)
        cnn.addLayer(layer_type = 'Conv', kernel_size = 3, num_kernels = 2, \
            activation = 'logistic', weights = np.append(
                np.concatenate((l1k1.flatten(), l1b1)).reshape(1,np.concatenate((l1k1.flatten(), l1b1)).shape[0]),
                np.concatenate((l1k2.flatten(), l1b2)).reshape(1,np.concatenate((l1k2.flatten(), l1b1)).shape[0]),
                axis=0))
        cnn.addLayer(layer_type = 'Pool', kernel_size = 2)
        cnn.addLayer(layer_type = 'Flatten')
        cnn.addLayer(layer_type = 'FC', num_neurons = 1, activation = 'logistic', \
            weights = np.concatenate((l3w.flatten(), l3b)).reshape(1,np.concatenate((l3w.flatten(), l3b)).shape[0]))

        np.set_printoptions(precision = 5)

        input = np.reshape(input, (1, 8, 8))

        print('model output before:')
        print(cnn.calculate(input))

        # Train the network
        cnn.train(input, output)
        out = cnn.calculate(input)
        print('model output after:')
        print(out)

        np.set_printoptions(precision=5)
        # Get weights:
        kernel1 = cnn.network[0].kernels[0][0].w
        print('1st convolutional layer, 1st kernel weights:')
        print(kernel1[:-1].reshape((3, 3)))
        print('1st convolutional layer, 1st kernel bias:')
        print(kernel1[-1])

        kernel2 = cnn.network[0].kernels[1][0].w
        print('1st convolutional layer, 2nd kernel weights:')
        print(kernel2[:-1].reshape((3, 3)))
        print('1st convolutional layer, 2nd kernel bias:')
        print(kernel2[-1])

        fc = cnn.network[-1].neurons[0].w
        print('fully connected layer weights:')
        print(fc[:-1])
        print('fully connected layer bias:')
        print(fc[-1])

if __name__ == '__main__':
    # Options for running the code
    if len(sys.argv) < 3:
        print("usage: python3 run_proj2.py <example number> <'keras' or 'project'>")

    elif sys.argv[1] == '1':
        run_example1(sys.argv[2])

    elif sys.argv[1] == '2':
        run_example2(sys.argv[2])

    elif sys.argv[1] == '3':
        run_example3(sys.argv[2])
