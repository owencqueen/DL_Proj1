import numpy as np

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

def get_train(fname):
    '''
    Gets training data from split file
    Accomplishes Task 2
    
    Arguments:
    ----------
    fname: string
        - Name of file that contains the split data

    Returns:
    --------
    None
    '''

    f = open(fname, 'r')
    lines = f.readlines()
    lines = [l.replace('\n', '') for l in lines]

    # Create map from keys to one-hot encoding
    onehot_map = {c:key for key, c in enumerate(sorted(list(set(''.join(lines)))))}
    vsize = len(onehot_map.keys())

    X = []
    for l in lines:
        X.append([])
        for c in l:
            X[-1].append(make_onehot(vsize, onehot_map[c]))

    # Leave out last sample (doesn't have next character for prediction)
    Xtrain = np.array(X)[:-1]
    print(Xtrain.shape)

    # Now get y labels:
    Y = []
    for i in range(len(X) - 1):
        # First character of i + 1 sequence
        Y.append(X[i + 1][0])

    Ytrain = np.array(Y)
    print(Ytrain.shape)

if __name__ == '__main__':
    #split_data('beatles.txt', 5, 3, write = False)
    get_train('lyrics_w=5_s=3.txt')


    