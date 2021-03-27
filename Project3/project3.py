import os, random
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

    mapping = {num:label for num, label in enumerate(le.classes_)}

    #print(numeric_labels)

    return numeric_labels, mapping

def plot_cm(model, Xval, Yval, val_map, title):
    '''
    Plot a confusion matrix for the given validation data

    Arguments:
    ----------
    model: keras.Model() object
        - Model that we're evaluating

    Returns:
    --------
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

    #plt.figure(fig_size = (10, 7))
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

    plot_cm(model, Xval, Yval, val_map, title = 'Task {}, Label = {} Validation Confusion Matrix'.format(args['task_number'], label))

def evoke_task(task_number = 'task1', label = 'gender'):
    '''
    Evokes the given task number from the command line
        - Parsing done in if main statement
    
    Arguments:
    ----------

    Returns:
    --------
    '''

    if task_number == 'task1':

        if label == 'gender':
            l = 'binary_crossentropy'
            eps = 13
        elif label == 'age':
            l = tf.keras.losses.SparseCategoricalCrossentropy()
            eps = 11
        elif label == 'race':
            l = tf.keras.losses.SparseCategoricalCrossentropy()
            eps = 11
        
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
            eps = 13
        elif label == 'age':
            l = tf.keras.losses.SparseCategoricalCrossentropy()
            eps = 11
        elif label == 'race':
            l = tf.keras.losses.SparseCategoricalCrossentropy()
            eps = 11
        
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(filters=40, kernel_size=3, activation = 'relu', strides=1, input_shape = (32, 32, 1), padding = 'valid'))
        model.add(layers.Conv2D(filters=80, kernel_size=5, activation = 'relu', strides=1, padding = 'valid'))
        model.add(layers.MaxPooling2D(2))
        model.add(layers.Flatten())
        model.add(layers.Dense(1000, activation = 'relu'))

        args = {
            'loss': l,
            'optimizer': tf.keras.optimizers.SGD(learning_rate = 0.1),
            'epochs': eps,
            'batch_size': 128,
            'task_number': 3
        }

    train_model(model, args, label)

def task4(label = ['gender', 'age']):
    '''
    Runs task 4
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

    l = []
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
        elif i == 'age':
            l.append(tf.keras.losses.SparseCategoricalCrossentropy())
        elif i == 'race':
            l.append(tf.keras.losses.SparseCategoricalCrossentropy())
    
    eps = 30
    
    image_input = tf.keras.Input(shape=(32,32,1))
    h1 = layers.Conv2D(filters=40, kernel_size=3, activation = 'relu', strides=1, padding = 'valid')(image_input)
    h2 = layers.Conv2D(filters=80, kernel_size=5, activation = 'relu', strides=1, padding = 'valid')(h1)
    flatten = layers.Flatten()(h2)
    last_hidden = layers.Dense(1000, activation='relu')(flatten)

    output_layers = []
    for i in range(len(label)):
        if label[i] == 'gender':
            output_layers.append(layers.Dense(num_classes[i], name='gender')(last_hidden))
        elif label[i] == 'age':
            output_layers.append(layers.Dense(num_classes[i], name='age')(last_hidden))
        elif label[i] == 'race':
            output_layers.append(layers.Dense(num_classes[i], name='race')(last_hidden))


    model = keras.Model(inputs=image_input, outputs=output_layers)

    keras.utils.plot_model(model, "task4.png", show_shapes=True)

    # Using adam for all tasks
    model.compile(
        optimizer = tf.keras.optimizers.SGD(learning_rate = 0.1), 
        loss = l
    )

    # Fit the model
    history = model.fit(
        Xtrain, Ytrain, 
        validation_data = (Xval, Yval),
        epochs = eps,
        batch_size = 128
    )

    # Plot
    plt.plot(history.history['accuracy'], label = 'Training')
    plt.plot(history.history['val_accuracy'], label = 'Validation')
    plt.title('Task {} Training and Validation Accuracy'.format(4))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    for i in range(len(label)):
        plot_cm(model, Xval, Yval[i], val_map[i], title = 'Task {}, Label = {} Validation Confusion Matrix'.format(4, label[i]))

if __name__ == '__main__':
    evoke_task('task2', 'age')
    #a,b,c,d = load_data()
    #to_numeric(d['gender'])    
