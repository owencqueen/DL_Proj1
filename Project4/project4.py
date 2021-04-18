import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split

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

    return split_lines

def train_test(test_size, window, stride):
    '''
    Basically a wrapper on train_test_split to work with our system

    Arguments:
    ----------
    test_size: float
        - Size of testing split
    window: int
        - Same window parameter as split_data
    stride: int
        - Same stride parameter as split_data

    Returns:
    --------
    Xtrain, Xtest, Ytrain, Ytest
    Xtrain: ndarray
        - X training data
    Xtest: ndarray
        - X validation data
    Ytrain: ndarray
        - Y training data
    Ytest: ndarray
        - Y validation data
    '''

    # Get the lines without writing file
    lines = split_data('beatles.txt', window, stride, write = False)
    X, Y, onehot_to_char = get_train(lines, file = False)

    return train_test_split(X, Y, test_size = test_size, shuffle = True)

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

    Ytrain: 
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
    for l in lines:
        X.append([])
        for c in l:
            X[-1].append(make_onehot(vsize, onehot_map[c]))

    # Leave out last sample (doesn't have next character for prediction)
    Xtrain = np.array(X)[:-1]

    # Now get y labels:
    Y = []
    for i in range(len(X) - 1):
        # First character of i + 1 sequence
        Y.append(X[i + 1][0])

    Ytrain = np.array(Y)

    return Xtrain, Ytrain, onehot_to_char

def predict_char(initial_char, model, temp, num_char_pred, vocab_size):
    chars = initial_char
    generated_ix = []
    for i in range(num_char_pred):
        preds = model.predict(np.array([chars,]))[0]
        preds = np.log(preds)/temp
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        ix = np.argmax(probas)
        x = np.zeros((1, vocab_size))
        x[0][ix] = 1
        chars = np.append(chars, x, axis=0)
        chars = chars[1:]
        generated_ix.append(ix)
    return generated_ix

def train(model, X, Y, inverse_map, epochs=5):
    histories = []
    for e in range(1, epochs):
        history = model.fit(X, Y)
        histories.append(history)
        if (e % 1 == 0):
            ind = np.random.randint(0, len(X)-1)
            initial = X[ind]
            initial_ind = np.argmax(initial, axis=-1)
            txt = ''.join(inverse_map[ix] for ix in initial_ind)
            print ('\nInitial: {}'.format (txt))

            gen = predict_char(initial, model, 0.5, 100, X.shape[2])
            txt = ''.join(inverse_map[ix] for ix in gen)
            print ('----\n {} \n----'.format (txt))

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
    #val_loss = [h.history['val_loss'] for h in histories]

    plt.plot(range(0, len(train_loss)), train_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.show()

if __name__ == '__main__':
    #split_data('beatles.txt', 5, 3, write = False)
    X, Y, i_map = get_train('lyrics_w=5_s=3.txt')
    model = tf.keras.models.Sequential()
    model.add(layers.LSTM(5, input_shape=(6, 47)))
    model.add(layers.Dense(47, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam")

    h = train(model, X, Y, i_map)
    plot_loss_epoch(h)

    