import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras import optimizers

from project2 import *

def generateExample1():
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
    pass

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
        pass

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
        pass

def run_example3(option = 'keras'):
    '''
    Runs example 3
    Arguments:
    ----------
    option: string, optional
        - Default: 'keras'
        - Options: 'keras' or 'project'
    '''

    if option == 'keras':
        pass
    
    elif option == 'project':
        pass

if __name__ == '__main__':
    run_example1('keras')
