import os, random, sys
import numpy as np
import pandas as pd
import tensorflow as tf # Getting Tensorflow

from tensorflow import keras
from tensorflow.keras import layers

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

import seaborn as sns

import matplotlib.pyplot as plt

# Using Python Image Library (PIL) for image processing
from PIL import Image

# Import other module for task5
import vae

def load_data():
    '''
    Loads data from the given directories
        - Performs min-max scaling
    No arguments
    Returns:
    --------
    train, train_labels, val, val_labels
    train: list of (32, 32, 1) np arrays
        - All loaded pictures
        - Preserves sequence observed in each labels file
    train_labels: pd Dataframe
        - CSV-loaded file with labels
    val: same as train but for validation set
    val_labels: same as train_labels but for validation set
    '''
    train_labels = pd.read_csv('fairface_label_train.csv')
    val_labels = pd.read_csv('fairface_label_val.csv')

    # Load in training data:
    train = []
    for f in list(train_labels['file']):
        img = Image.open(f)
        train.append(np.reshape(np.array(img), (32, 32, 1)))

    val = []
    for f in list(val_labels['file']):
        img = Image.open(f)
        val.append(np.reshape(np.array(img), (32, 32, 1)))

    # Apply min-max scaling to both train and validation:
    train_max = np.array(train).max()
    train_min = np.array(train).min()

    train = (train - train_min) / (train_max - train_min)

    val_max = np.array(val).max()
    val_min = np.array(val).min()

    val = (val - val_min) / (val_max - val_min)

    return np.array(train), train_labels, np.array(val), val_labels

def to_numeric(pd_series):
    '''
    Converts series of labels into numeric with mapping
    
    Arguments:
    ----------
    pd_series: Pandas series or any array-lik
        - Series of labels to turn numeric

    Returns:
    --------
    numeric_labels: list
        - Labels in numeric form
    mapping: dictionary
        - Provides a mapping from values in numeric_labels to original label
    '''
    le = LabelEncoder()
    le.fit(pd_series)

    numeric_labels = le.fit_transform(pd_series)

    # Creates a mapping from numeric labels to their original, string label
    mapping = {num:label for num, label in enumerate(le.classes_)}

    return numeric_labels, mapping

def plot_cm(model, Xval, Yval, val_map, title):
    '''
    Plot a confusion matrix for the given validation data

    Arguments:
    ----------
    model: keras.Model() object
        - Model that we're evaluating
    Xval: ndarray
        - Validation data
    Yval: ndarray
        - Validation data
    val_map: dictionary
        - Map of numeric labels to original labels (to show on plot) 
    title: string
        - Title of given plot

    Returns:
    --------
    No return value
    '''
    preds = model.predict(Xval)

    num_classes = len(set(Yval))

    # Sigmoid decision rule:
    if num_classes == 2:
        ypred = [1 if p > 0.5 else 0 for p in preds]
    else:
        ypred = [np.argmax(p) for p in preds]

    cm = confusion_matrix(Yval, ypred)
    df_cm = pd.DataFrame(cm/np.sum(cm), index = [val_map[i] for i in range(len(set(Yval)))], 
                columns = [val_map[i] for i in range(len(set(Yval)))])

    plt.figure(figsize = (10, 7))
    sns.heatmap(df_cm, annot=True, fmt = '.2%')
    plt.title(title)
    plt.show()

def train_model(model, args, label = 'gender'):
    '''
    Runs task 1, 2, or 3 from writeup
    Note: this function does not perform advanced preprocessing other than loading data
    Generates two plots:
        1) Plot of validation and training accuracy
        2) Confusion matrix (not tested for >2 classes)

    Arguments:
    ----------
    model: keras model instance
        - Built model (not trained)
    args: dictionary
        - Should contain values with keys:
        'optimizer', 'loss', 'epochs', 'batch_size', 'task_number'
    label: string, optional
        - Default: 'gender'
        - Label you're trying to predict

    Returns:
    --------
    No return value
    '''

    Xtrain, train_labels, Xval, val_labels = load_data()

    # Create mappings for Y values
    Ytrain, train_map = to_numeric(train_labels[label])
    Yval, val_map = to_numeric(val_labels[label])

    num_classes = train_labels[label].value_counts().shape[0]

    if num_classes == 2:
        model.add(layers.Dense(1, activation = 'sigmoid'))
    else:
        model.add(layers.Dense(num_classes, activation = 'softmax'))

    # Using adam for all tasks
    model.compile(
        optimizer = args['optimizer'], 
        loss = args['loss'], metrics = ['accuracy']
    )

    print(model.summary())

    keras.utils.plot_model(model, str(args['task_number'])+'_'+label+'.png', show_shapes=True)

    # Fit the model
    history = model.fit(
        Xtrain, Ytrain, 
        validation_data = (Xval, Yval),
        epochs = args['epochs'],
        batch_size = args['batch_size']
    )

    # Plot
    plt.plot(history.history['accuracy'], label = 'Training')
    plt.plot(history.history['val_accuracy'], label = 'Validation')
    plt.title('Task {} Training and Validation Accuracy'.format(args['task_number']))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plot_cm(model, Xval, Yval, val_map, title = 'Task {}, {} Validation Confusion Matrix'.format(args['task_number'], label))

def evoke_task(task_number = 'task1', label = 'gender'):
    '''Evokes the given task number from the command line
        - Parsing done in if main statement
    Arguments:
    ----------
    task_number: string, optional
        - Default: 'task1'
        - 'task1', 'task2', or 'task3'
        - Can be the direct input from command line (based on cmd line specs)
    label: string, optional
        - Default: 'gender'
        - 'gender', 'race', or 'age'

    Returns:
    --------

    '''

    if task_number == 'task1':

        if label == 'gender':
            l = 'binary_crossentropy'
            eps = 50
        elif label == 'age':
            l = tf.keras.losses.SparseCategoricalCrossentropy()
            eps = 50
        elif label == 'race':
            l = tf.keras.losses.SparseCategoricalCrossentropy()
            eps = 50
        
        model = tf.keras.Sequential()
        model.add(keras.Input(shape=(32,32, 1)))
        model.add(layers.Flatten())
        model.add(layers.Dense(1024, activation='tanh'))
        model.add(layers.Dense(512, activation='sigmoid'))
        model.add(layers.Dense(100, activation='relu'))

        args = {
            'loss': l,
            'optimizer': tf.keras.optimizers.SGD(learning_rate = 0.1),
            'epochs': eps,
            'batch_size': 128,
            'task_number': 1
        }

    if task_number == 'task2':

        if label == 'gender':
            l = 'binary_crossentropy'
            eps = 13
        elif label == 'age':
            l = tf.keras.losses.SparseCategoricalCrossentropy()
            eps = 11

        # Note: default is strides=1
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(filters=40, kernel_size=5, activation = 'relu', strides=1, input_shape = (32, 32, 1), padding = 'valid'))
        model.add(layers.MaxPooling2D(2))
        model.add(layers.Flatten())
        model.add(layers.Dense(100, activation = 'relu'))
        # Note: don't include activation layer, train_model does it for you

        args = {
            'loss': l,
            'optimizer': tf.keras.optimizers.SGD(learning_rate = 0.1),
            'epochs': eps,
            'batch_size': 128,
            'task_number': 2
        }

    if task_number == 'task3':

        if label == 'gender':
            l = 'binary_crossentropy'
            eps = 50
            lr = 0.01
        elif label == 'age':
            l = tf.keras.losses.SparseCategoricalCrossentropy()
            eps = 50
            lr = 0.1
        elif label == 'race':
            l = tf.keras.losses.SparseCategoricalCrossentropy()
            eps = 50
            lr = 0.01
        
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(filters=64, kernel_size=3, activation = 'relu', strides=1, input_shape = (32, 32, 1), padding = 'same'))
        model.add(layers.Conv2D(filters=64, kernel_size=3, activation = 'relu', strides=1, padding = 'same'))
        model.add(layers.Conv2D(filters=64, kernel_size=3, activation = 'relu', strides=1, padding = 'same'))
        model.add(layers.MaxPooling2D(2))
        #model.add(layers.Dropout(rate=0.5))
        model.add(layers.Conv2D(filters=128, kernel_size=3, activation = 'relu', strides=1, padding = 'same'))
        model.add(layers.Conv2D(filters=128, kernel_size=3, activation = 'relu', strides=1, padding = 'same'))
        model.add(layers.Conv2D(filters=128, kernel_size=3, activation = 'relu', strides=1, padding = 'same'))
        model.add(layers.MaxPooling2D(2))
        model.add(layers.Dropout(rate=0.5))
        model.add(layers.Conv2D(filters=256, kernel_size=2, activation = 'relu', strides=1, padding = 'same'))
        model.add(layers.Conv2D(filters=256, kernel_size=2, activation = 'relu', strides=1, padding = 'same'))
        model.add(layers.Conv2D(filters=256, kernel_size=2, activation = 'relu', strides=1, padding = 'same'))
        model.add(layers.MaxPooling2D(2))
        #model.add(layers.Dropout(rate=0.5))
        model.add(layers.Flatten())
        model.add(layers.Dense(4096, activation = 'relu'))
        model.add(layers.Dense(1024, activation = 'relu'))
        model.add(layers.Dense(1024, activation = 'relu'))

        args = {
            'loss': l,
            'optimizer': tf.keras.optimizers.SGD(learning_rate = lr),
            'epochs': 50,
            'batch_size': 128,
            'task_number': 3
        }
    
    if task_number == 'task4':
        task4()

    if task_number == 'task5':
        args = {
        'input_shape':(32, 32, 1), 
        'bottleneck_size':15, 
        'reshape_size':(16, 16, 32), 
        'batch_size': 128,
        'epochs':10  
        }
        vae.run_vae('gender', args, save = False)
        return 0

    train_model(model, args, label)

def task4(label=['gender','age']):
    '''
    Runs task 4
    Note: this function does not perform advanced preprocessing other than loading data
    Generates two plots:
        1) Plot of validation and training accuracy
        2) Confusion matrix (not tested for >2 classes)

    Arguments:
    ----------
    label: list
        - Default: ['gender', 'age']
        - Labels you're trying to predict

    Returns:
    --------
    No return value

    '''

    Xtrain, train_labels, Xval, val_labels = load_data()

    l = []
    l_wts = []
    Ytrain = []
    Yval = []
    num_classes = []
    val_map = []
    for i in label:
        tr_dat, tr_map = to_numeric(train_labels[i])
        Ytrain.append(tr_dat)
        val_dat, val_m = to_numeric(val_labels[i])
        val_map.append(val_m)
        num_classes.append(train_labels[i].value_counts().shape[0])
        Yval.append(val_dat)
        if i == 'gender':
            l.append('binary_crossentropy')
            l_wts.append(0.1)
        elif i == 'age':
            l.append(tf.keras.losses.SparseCategoricalCrossentropy())
            l_wts.append(4.0)
        elif i == 'race':
            l.append(tf.keras.losses.SparseCategoricalCrossentropy())
            l_wts.append(1.0)
    
    eps = 50
    
    image_input = tf.keras.Input(shape=(32,32,1))
    c1 = layers.Conv2D(filters=64, kernel_size=3, activation = 'relu', strides=1, padding = 'same')(image_input)
    c2 = layers.Conv2D(filters=64, kernel_size=3, activation = 'relu', strides=1, padding = 'same')(c1)
    c3 = layers.Conv2D(filters=64, kernel_size=3, activation = 'relu', strides=1, padding = 'same')(c2)
    bn1 = layers.BatchNormalization()(c3)
    m1 = layers.MaxPooling2D(2)(bn1)
    d1 = layers.Dropout(rate=0.25)(m1)
    c4 = layers.Conv2D(filters=128, kernel_size=3, activation = 'relu', strides=1, padding = 'same')(d1)
    c5 = layers.Conv2D(filters=128, kernel_size=3, activation = 'relu', strides=1, padding = 'same')(c4)
    c6 = layers.Conv2D(filters=128, kernel_size=3, activation = 'relu', strides=1, padding = 'same')(c5)
    bn2 = layers.BatchNormalization()(c6)
    m2 = layers.MaxPooling2D(2)(bn2)
    d2 = layers.Dropout(rate=0.25)(m2)
    c7 = layers.Conv2D(filters=256, kernel_size=2, activation = 'relu', strides=1, padding = 'same')(d2)
    c8 = layers.Conv2D(filters=256, kernel_size=2, activation = 'relu', strides=1, padding = 'same')(c7)
    c9 = layers.Conv2D(filters=256, kernel_size=2, activation = 'relu', strides=1, padding = 'same')(c8)
    m3 = layers.MaxPooling2D(2)(c9)
    d3 = layers.Dropout(rate=0.25)(m3)
    flatten = layers.Flatten()(d3)

    output_layers = []
    for i in range(len(label)):
        if label[i] == 'gender':
          g1 = layers.Dense(4096, activation='relu')(flatten)
          g2 = layers.Dense(2048, activation='relu')(g1)
          
          output_layers.append(layers.Dense(num_classes[i]-1, name='gender', activation='sigmoid')(g2))
        elif label[i] == 'age':
          a1 = layers.Dense(4096, activation='relu')(flatten)
          a2 = layers.Dense(2048, activation='relu')(a1)
          
          output_layers.append(layers.Dense(num_classes[i], name='age', activation='softmax')(a2))
        elif label[i] == 'race':
          r1 = layers.Dense(4096, activation='relu')(flatten)
          r2 = layers.Dense(2048, activation='relu')(r1)
          output_layers.append(layers.Dense(num_classes[i], name='race', activation='softmax')(r2))


    model = keras.Model(inputs=image_input, outputs=output_layers)

    # Using adam for all tasks
    model.compile(
        optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01),#, decay=0.1/eps), 
        loss = l,
        metrics=["accuracy"],
        #loss_weights = l_wts
    )

    # Fit the model
    history = model.fit(
        Xtrain, Ytrain, 
        validation_data = (Xval, Yval),
        epochs = eps,
        batch_size = 128
    )

    (fig, ax) = plt.subplots(len(label), 1, figsize=(8, 8))
    fig.subplots_adjust(hspace=0.5)
    for (i, l) in enumerate(label):
      # plot the loss for both the training and validation data
      ax[i].set_title("Accuracy for {}".format(l))
      ax[i].set_xlabel("Epoch #")
      ax[i].set_ylabel("Accuracy")
      ax[i].plot(np.arange(0, eps), history.history[l+'_accuracy'], label='Training')
      ax[i].plot(np.arange(0, eps), history.history["val_" + l+'_accuracy'],
        label='Validation')
      ax[i].legend()
    plt.show()

    preds = model.predict(Xval)
    for i in range(len(label)):

      num_classes = len(set(Yval[i]))

      # Sigmoid decision rule:
      if num_classes == 2:
          ypred = [1 if p > 0.5 else 0 for p in preds[i]]
      else:
          ypred = [np.argmax(p) for p in preds[i]]

      cm = confusion_matrix(Yval[i], ypred)
      df_cm = pd.DataFrame(cm/np.sum(cm), index = [val_map[i][j] for j in range(len(set(Yval[i])))], 
                  columns = [val_map[i][j] for j in range(len(set(Yval[i])))])

      plt.figure(figsize = (10, 7))
      sns.heatmap(df_cm, annot=True, fmt = '.2%')
      plt.title('Task {}, {} Validation Confusion Matrix'.format(4, label[i]))
      plt.show()

if __name__ == '__main__':
    options = set(['task' + str(i) for i in range(1, 6)])
    label_options = set(['gender', 'age'])

    if len(sys.argv) < 2:
        print('usage: python3 project3.py task<1-5> <gender or age>')
        exit()

    # Command line interface:
    if (str(sys.argv[1]) not in options):
        print('usage: python3 project3.py task<1-5> <gender or age>')
        exit()
    
    if (str(sys.argv[1]) != 'task5' and str(sys.argv[1]) != 'task4'):
        if len(sys.argv) < 3:
            print('usage: python3 project3.py task<1-5> <gender or age>')
            exit()

        elif (str(sys.argv[2]) not in label_options):
            print('usage: python3 project3.py task<1-5> <gender or age>')
            exit()
        next_arg = sys.argv[2]
    else:
        next_arg = None

    evoke_task(sys.argv[1], next_arg)
