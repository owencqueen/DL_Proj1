# Owen Queen and Sai Thatigotla: Project 4, COSC 525

import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, SimpleRNN

import matplotlib.pyplot as plt

def split_data(fname, window, stride, write = False):
    '''
    Splits the text file by window and stride size
    Accomplishes Task 1
    
    Arguments:
    ----------
    fname: string
        - Name of file to read from
    window: int
        - Size of window to use when computing training data
    stride: int
        - Stride by which to increment on each window
    write: bool, optional
        - If True, writes the training data to file
    Returns:
    --------
    split_lines: list of strings
        - All lines that are written to the file
    '''

    f = open(fname, 'r') # Open read-only file
    lines = f.read().replace('\n',' ')

    split_lines = []

    # Does the window-stride operation
    i = 0
    while (i + window) < len(lines):
        split_lines.append(lines[i:(i + window + 1)])
        i += stride

    if write: # writes the files
        wf = open('lyrics_w={}_s={}.txt'.format(window, stride), 'w')
        # Consistent formatting prevents writing of files storing the same data
        full_txt = '\n'.join(split_lines)
        wf.write(full_txt)
        wf.close()

    f.close()

    return split_lines, lines

def make_onehot(vsize, ind):
    '''
    Makes a one-hot encoding for a character
    Arguments:
    ----------
    vsize: int
        - Size of vocabulary
        - Determines size of array in output
    ind: int
        - Index that will be marked 1
    Returns:
    --------
    g: ndarray of size (vsize,)
        - One-hot encoded array
    '''
    g = np.zeros(vsize)
    g[ind] = 1.0
    return g

def get_train(fname, file = True):
    '''
    Gets training data from split file
    Accomplishes Task 2
    
    Arguments:
    ----------
    fname: string
        - Name of file that contains the split data
    Returns:
    --------
    Xtrain: (m, n, p) ndarray
        - m: number of sequences
        - n: length of sequences
        - p: vocabulary size
    Ytrain: (m, p) ndarray
        - m: number of sequences
        - p: vocabulary size
    onehot_map: dict
        - Mapping from char to one-hot encoding index
    onehot_to_char: dict
        - Mapping from one-hot encoding index to char
    '''

    if file:
        f = open(fname, 'r')
        lines = f.readlines()
        lines = [l.replace('\n', '') for l in lines]
    else:
        lines = fname

    # Create map from keys to one-hot encoding
    onehot_map = {c:key for key, c in enumerate(sorted(list(set(''.join(lines)))))}
    onehot_to_char = {key:c for key, c in enumerate(sorted(list(set(''.join(lines)))))}

    vsize = len(onehot_map.keys())

    X = []
    Y = []
    for l in lines:
        X.append([])
        Y.append(make_onehot(vsize, onehot_map[l[-1]]))
        for c in l[:-1]:
            X[-1].append(make_onehot(vsize, onehot_map[c]))

    # Leave out last sample (doesn't have next character for prediction)
    Xtrain = np.array(X)
    Ytrain = np.array(Y)

    return Xtrain, Ytrain, onehot_map, onehot_to_char

def predict_char(initial_char, model, temp, num_char_pred, vocab_size, window_size, orig_map, inverse_map):
    '''
    Arguments:
    ----------
    intial_char: ndarray
        - Initial string
    model: keras Model object
        - Trained model which we use to make predictions
    temp: float
        - Sampling temperature
    num_char_pred: int
        - Number of characters that we wish to predict
    vocab_size: int
        - Size of vocabulary for entire training/prediction problem
    window_size: int
        - Size of window used in preprocessing
    orig_map: dict
        - Mapping from character to index for one-hot encoding
    inverse_map: dict
        - Reverse mapping of orig_map

    Returns:
    --------
    generated_chars: string
        - Generated characters predicted by the model
    '''
    chars = initial_char
    
    generated_chars = ""
    
    for i in range(num_char_pred):
        input_chars = np.zeros((1, window_size, vocab_size))
        
        for j,k in enumerate(chars):
          input_chars[0, j, orig_map[k]] = 1.0

        preds = model.predict(np.array(input_chars))
        preds = preds[0]
        preds = np.asarray(preds).astype('float64')

        # Temp/Softmax on predictions:
        preds = np.log(preds)/temp
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)

        # Sampling based on predictions:
        probas = np.random.multinomial(1, preds, 1)

        ix = np.argmax(probas)
        next_char = inverse_map[ix]
        
        # Increment strings:
        chars += next_char
        generated_chars += next_char

        chars = chars[1:]
        
    return generated_chars

def train(model, X, Y, orig_map, inverse_map, lines, epochs=100, temp=[0.01, 0.25, 0.5, 0.75, 1.0]):
    '''
    Arguments:
    ----------
    model: keras Model object
        - model object holding architecture to be trained
    X: nd array
        - Training data in the form of a tensor
    Y: nd array
        - 
    orig_map: dict
        - Mapping from character to index for one-hot encoding
    inverse_map: dict
        - Reverse mapping of orig_map
    lines: list of strings
        - Separated strings that represent the training file broken into window, stride
    epochs: int
        - Number of epochs to train model
    temp: list of floats OR int, optional
        - Default: [0.01, 0.25, 0.5, 0.75, 1.0]
        - Sampling temperatures to use for a qualitative evaluation of the model at every fourth epoch

    Returns:
    --------
    histories: list of History objects
        - Histories from each epoch that the model is ran
    '''

    histories = []
    for e in range(1, epochs+1):
        history = model.fit(X, Y, batch_size=64)
        histories.append(history)

        # Evaluate every 4th epoch:
        if ((e % 4 == 0) or (e == epochs)):
            
            ind = np.random.randint(0, len(lines) - X.shape[1] - 1)
            
            initial = lines[ind: ind+X.shape[1]]

            print('Initial: {}\n'.format(initial))

            # Qualitative evaulation on predictions based on random sequence
            if (isinstance(temp, list)):
                for j in temp:
                    gen = predict_char(initial, model, j, 100, X.shape[2], X.shape[1], orig_map, inverse_map)
                    print ('----\nTemperature: {}\n{} \n----'.format (j, gen))

            else:
                gen = predict_char(initial, model, temp, 100, X.shape[2], X.shape[1], orig_map, inverse_map)
                txt = ''.join(inverse_map[ix] for ix in gen)
                print ('----\nTemperature: {}\n{} \n----'.format (temp, txt))

    return histories

def plot_loss_epoch(histories, title = ''):
    '''
    Plots loss vs. epoch

    Arguments:
    ----------
    histories: list of History objects
        - Should be return from train function
    title: string, optional
        - Default: '' - i.e. no title
        - Title to show on plot

    Returns:
    --------
    None
    '''

    train_loss = [h.history['loss'] for h in histories]

    plt.plot(range(0, len(train_loss)), train_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.show()

if __name__ == '__main__': 
    # Command line interface for the project

    model_opts = {'lstm', 'rnn'}
    
    # Error checking
    if len(sys.argv) != 7:
        print('usage: python3 project4.py <file> <lstm or rnn> <hidden state size> <window size> <stride> <temperature>')
        exit()
    
    if not (sys.argv[2] in model_opts):
        print('usage: python3 project4.py <file> <lstm or rnn> <hidden state size> <window size> <stride> <temperature>')
        exit()
    
    # Setting up arguments
    hstate = int(sys.argv[3])
    window = int(sys.argv[4])
    stride = int(sys.argv[5])
    temp = float(sys.argv[6])

    splits, lines = split_data(sys.argv[1], window, stride, write = False)
    X, Y, orig_map, i_map = get_train(splits, file = False)

    vocab_size = len(list(orig_map.keys()))

    # Builds the given model architectures
    model = tf.keras.models.Sequential()
    if sys.argv[2] == 'lstm':
        model.add(layers.LSTM(hstate, input_shape = (window, vocab_size)))

    elif sys.argv[2] == 'rnn':
        model.add(layers.SimpleRNN(hstate, input_shape = (window, vocab_size)))

    model.add(layers.Dense(vocab_size, activation = 'softmax'))

    # Compiles the model
    model.compile(loss='categorical_crossentropy', optimizer = 'adam')

    # Train the model and show the loss plot
    h = train(model, X, Y, orig_map, i_map, lines, epochs = 15)
    plot_loss_epoch(h, title = sys.argv[2].upper() + ' w = {}, stride = {}, hidden units = {} Loss'.format(window, stride, hstate))
