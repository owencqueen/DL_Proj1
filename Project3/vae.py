import numpy as np

import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras import backend

from project3 import load_data, to_numeric

import matplotlib.pyplot as plt

'''
Variation Autoencoder portion of Project 3
- Author: Owen Queen

Used the following sources to write the code:
    1. Dr. Sadovnik's 'VAE.html' notebook
    2. "Building Autoencoders in Keras" post on "The Keras Blog"
    3. Keras tutorial: https://keras.io/examples/generative/vae/

Note before running:
    - I trained this model on GPUs on Google Colab, so it may take quite a while
        if you are not running this on GPUs
'''

def vae_sampling(args):
    '''
    Performs sampling for bottleneck layer in VAE
        - Does reparameterization trick
    Note: Adapted from Dr. Sadovnik's Code on Canvas

    Arguments:
    ----------
    args: tensor
        - Mean and log(variance) of Q(z|X) (conditional dist. given X input)

    Returns:
    --------
    sample_value: float
        - Value after sampling from distribution in bottleneck layer
    '''

    # Get mean and variance from arguments
    mean, log_var = args

    # Get batch and dim
    batch = backend.shape(mean)[0]
    dim = backend.int_shape(mean)[1]

    # Samples a value of epsilon to perturb sampling
    ep = backend.random_normal(shape = (batch, dim))

    return mean + backend.exp(log_var) * ep

def build_model(args):
    '''
    Sets up the architecture for the encoder and decoder
    Internally used in the vae_model class

    Arguments:
    ----------
    args: dictionary
        - Need the following elements:
            1. 'input_shape': tuple; size of input
            2. 'bottleneck_size': int; size of bottleneck layer

    Returns:
    --------
    model: keras.Model object
        - End-to-end variational autoencoder
        - Please don't use this model, use the vae_model class to instantiate a model
    encoder: keras.Model object
        - Encoder with the architecture shown below
    decoder: keras.Model object
        - Decoder with the architecture shown below
    '''
    inputs = tf.keras.layers.Input(shape = args['input_shape'], name = 'encode_input')

    # Two convolutional layers in encoder:
    encode1 = tf.keras.layers.Conv2D(16, 3, activation = 'relu', name = 'encode1', padding = 'same')(inputs)
    encode2 = tf.keras.layers.Conv2D(32, 3, activation = 'relu', name = 'encode2', padding = 'same')(encode1)
    pool_encode = tf.keras.layers.MaxPooling2D((2, 2), name = 'encode_pool')(encode2)
    flat = tf.keras.layers.Flatten()(pool_encode)
    
    # Bottleneck layers
    bottleneck_mu = tf.keras.layers.Dense(args['bottleneck_size'], name = 'mu')(flat)
    bottleneck_sigma = tf.keras.layers.Dense(args['bottleneck_size'], name = 'log_var')(flat)

    # Sampler layer (w/ reparameterization trick)
    sampler = tf.keras.layers.Lambda(vae_sampling, name = 'z')([bottleneck_mu, bottleneck_sigma])

    # Full encoder model:
    encoder = Model(inputs, [bottleneck_mu, bottleneck_sigma, sampler], name = 'encoder')

    #print('ENCODE SHAPE:', encoder.get_layer('encode_pool').output_shape)

    '''Need: 'bottleneck_size' and 'reshape_size' '''
    bottleneck_inputs = tf.keras.layers.Input(shape=(args['bottleneck_size'],), name = 'decode_input')
    x = tf.keras.layers.Dense(np.prod(encoder.get_layer('encode_pool').output_shape[1:]), activation = 'relu')(bottleneck_inputs)
    x = tf.keras.layers.Reshape(encoder.get_layer('encode_pool').output_shape[1:])(x)
    x = tf.keras.layers.UpSampling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(32, 3, activation = 'relu', name = 'decode1', padding = 'same')(x)
    x = tf.keras.layers.Conv2D(16, 3, activation = 'relu', name = 'decode2', padding = 'same')(x)
    decoded = tf.keras.layers.Conv2D(1, 3, activation = 'relu', name = 'decode3', padding = 'same')(x)

    # Full decoder model:
    decoder = Model(bottleneck_inputs, decoded, name = 'decoder_output')

    outputs = decoder(encoder(inputs)[-1])

    model = Model(inputs, outputs, name = 'vae')

    return model, encoder, decoder

class vae_model(tf.keras.Model):
    ''' 
    Define custom vae class 
    Adapted from methods in: https://keras.io/examples/generative/vae/
    '''
    def __init__(self, encoder, decoder, **kwargs):
        '''
        Arguments:
        ----------
        encoder: tf.keras.Model object
            - Model object whose architecture has already been built
            - Serves as encoder in the VAE
        decoder:
            - Model object whose architecture has already been built
            - Serves as decoder in the VAE
        **kwargs: any other arguments that could be passed to tf.keras.Model __init__

        Returns:
        --------
        No return
        '''
        super(vae_model, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

        # Tracking various losses:
        self.total_loss_track = tf.keras.metrics.Mean(name = 'tot_loss')
        self.reconstruct_loss_track = tf.keras.metrics.Mean(name = 'reconstruct_loss')
        self.val_loss_track = tf.keras.metrics.Mean(name = 'val_loss')
        self.val_reconstruct_loss_track = tf.keras.metrics.Mean(name = 'reconstruct_val_loss')

    '''
    The rest of the functions are needed specifically for the training of the Keras model
        - They have been modified specifically to train the VAE
    '''

    @property
    def metrics(self):
        # Metrics that are returned with every successive batch
        return [self.total_loss_track, self.reconstruct_loss_track]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            mu, log_var, z = self.encoder(data)
            reconstruct = self.decoder(z)
            reconstruct_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.mse(data, reconstruct), axis = (1, 2)
                )
            )
            kl_loss = (backend.exp(log_var) + backend.square(mu) - 1)
            kl_loss = 0.001 * tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = backend.mean(reconstruct_loss + kl_loss)

        # Manual updating of gradients:
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        # Update the loss trackers:
        self.total_loss_track.update_state(total_loss)
        self.reconstruct_loss_track.update_state(reconstruct_loss)

        return {
            'loss':self.total_loss_track.result(),
            'reconstruct_loss': self.reconstruct_loss_track.result()
        }

    def call(self, inputs):
        # Reconstructs the input
        mu, log_var, z = self.encoder(inputs)
        reconstruct = self.decoder(z)
        return reconstruct

    def test_step(self, data):

        # Gets fed in data (standard set by inherited keras.Model class)
        x, y = data
        mu, log_var, z = self.encoder(x)

        reconstruct = self.decoder(z)

        # Calculate validation loss
        reconstruct_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.mse(x, reconstruct), axis = (1, 2)
            )
        )
        kl_loss = (backend.exp(log_var) + backend.square(mu) - 1)
        kl_loss = 0.001 * tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = backend.mean(reconstruct_loss + kl_loss)

        # Update validation loss counters
        self.val_loss_track.update_state(total_loss)
        self.val_reconstruct_loss_track.update_state(reconstruct_loss)

        # Return the losses with key so they'll show up in our verbose training routine
        return {
            'loss': total_loss,
            'reconstruct_loss': reconstruct_loss,
        }

def generate_data_linear(decoder, latent_dim, mu = 0, std = 1, save = False):
    '''
    Generates random samples to demonstrate the learned decoder

    Arguments:
    ----------
    decoder: keras.Model object
        - Trained decoder
    latent_dim: int
        - Size of the embedding layer (bottleneck)
        - Determines the dimensionality of the vectors that we sample to feed into decoder
    mu: float, optional
        - Default: 0
        - Mean of the normal distribution from which we draw our samples
    std: float, optional
        - Default: 1
        - Standard deviation of the normal distribution from which we draw our samples
    save: bool, optional
        - Default: False
        - If true, saves the generated image under 'generated_data.png' 
            in current working directory

    Returns:
    --------
    No return
    '''

    fig, ax = plt.subplots(2, 5, figsize=(20,10))

    # Generate 10 samples of random vecs
    for i in [0, 1]:
        for j in range(5):
            samp = np.random.normal(loc = mu, scale = std, size = latent_dim)
            #samp = samp.reshape((samp.shape[0], 1))
            samp = np.array([list(samp)])
            img = decoder.predict(samp)
            ax[i][j].imshow(np.reshape(img, (32, 32)), cmap = 'Greys_r')

    fig.suptitle('Sampled Images from VAE')

    if save:
      plt.savefig('generated_data.png')

    plt.show()


def run_vae(label, args, save = False):
    '''
    Runs the variational autoencoder
        - Most options have been hard-coded

    Arguments:
    ----------
    label: string
        - Key for Y vector in the csv files
    args: dictionary
        - Need the following arguments:
            1. 'input_shape': tuple; size of input to encoder
            2. 'bottleneck_size': int; size of bottleneck layer
            3. 'reshape_size': tuple; size to use 
            4. 'batch_size': int; batch size
            5. 'epochs': int; number of epochs to train
    save: bool, optional
        - Default: False
        - If True, saves the loss plot and generated images to your local directory

    Returns:
    --------
    No return value
    '''

    # Load data:
    Xtrain, train_labels, Xval, val_labels = load_data()

    Ytrain, train_map = to_numeric(train_labels[label])
    Yval, val_map = to_numeric(val_labels[label])

    vae, encoder, decoder = build_model(args)

    vae = vae_model(encoder, decoder)

    # Training:
    vae.compile(optimizer = 'adam')
    history = vae.fit(Xtrain, epochs = args['epochs'], batch_size = args['batch_size'], validation_data = (Xval, Yval))
    
    # Plot the loss curve
    plt.figure(figsize=(20, 10))
    plt.plot(history.history['loss'], label = 'Training')
    plt.plot(history.history['val_loss'], label = 'Validation')
    plt.title('Task 5 Training and Validation Loss (MSE + KL)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    if save:
      plt.savefig('task5_loss_curves.png')

    plt.show()

    # Generate example images
    generate_data_linear(vae.decoder, args['bottleneck_size'], mu=0, std=4, save = save)

if __name__ == '__main__':
    args = {
        'input_shape':(32, 32, 1), 
        'bottleneck_size':15, 
        'reshape_size':(16, 16, 32), 
        'batch_size': 128,
        'epochs':10  
    }

    run_vae('gender', args, save = True)