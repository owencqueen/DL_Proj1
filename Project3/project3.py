import os, random
import numpy as np
import pandas as pd
import tensorflow as tf # Getting Tensorflow

from tensorflow import keras
from tensorflow.keras import layers

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

# Using Python Image Library (PIL) for image processing
from PIL import Image

def load_data():
    '''Loads data from the given directories'''
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

    '''
    fig, axs = plt.subplots(1, 4)
    for i in range(4):
        axs[i].imshow(train[random.sample(range(len(train)), 1)].reshape((32, 32)))

    #plt.imshow(train[0].reshape((32, 32)))
    plt.show()
    '''

    #print('Train labels', train_labels['gender'])
    #print('val labels', val_labels['gender'])

    return np.array(train), train_labels, np.array(val), val_labels

def to_numeric(pd_series):
    '''Converts series of labels into numeric with mapping'''
    le = LabelEncoder()
    le.fit(pd_series)

    numeric_labels = le.fit_transform(pd_series)

    mapping = {num:label for num, label in enumerate(le.classes_)}

    #print(numeric_labels)

    return numeric_labels, mapping


def train_model(model, args, label = 'gender'):
    '''Runs task 1, 2, or 3 from writeup
    Note: this function does not perform advanced preprocessing other than loading data'''

    Xtrain, train_labels, Xval, val_labels = load_data()

    #Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], 32, 32, 1))

    Ytrain, train_map = to_numeric(train_labels[label])
    #print(Ytrain.shape)
    #print(Xtrain.shape)
    
    Yval, val_map = to_numeric(val_labels[label])
    #print('y val', Yval.shape)
    #print(Xval.shape)

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

    #print('Xtrain', Xtrain)
    #print(Xtrain)
    #print('Y', Ytrain)
    #print(args.items())

    history = model.fit(
        Xtrain, Ytrain, 
        validation_data = (Xval, Yval),
        epochs = args['epochs'],
        batch_size = args['batch_size']
    )

    plt.plot(history.history['loss'], label = 'Training')
    plt.plot(history.history['val_loss'], label = 'Validation')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def evoke_task(task_number = 'task1', label = 'gender'):

    if task_number == 'task1':
        pass

    if task_number == 'task2':

        if label == 'gender':
            l = 'binary_crossentropy'
        else:
            l = tf.keras.losses.CategoricalCrossentropy()

        # Note: default is strides=1
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(filters=40, kernel_size=5, activation = 'relu', strides=1, input_shape = (32, 32, 1), padding = 'valid'))
        model.add(layers.MaxPooling2D(2))
        model.add(layers.Flatten())
        model.add(layers.Dense(100, activation = 'relu'))
        # Note: don't include activation layer, train_model does it for you

        args = {
            'loss': l,
            'optimizer': 'sgd',#tf.keras.optimizers.Adam(learning_rate = 0.1),
            'epochs': 3,
            'batch_size': 128
        }

        train_model(model, args, label)

if __name__ == '__main__':
    evoke_task('task2', 'gender')
    #a,b,c,d = load_data()
    #to_numeric(d['gender'])    
