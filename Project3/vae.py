import numpy as np

import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras import backend

from project3 import load_data, to_numeric

import matplotlib.pyplot as plt

'''
Variation Autoencoder portion of the project
Used the following sources to write the code:
    1. Dr. Sadovnik's 'VAE.html' notebook
    2. "Building Autoencoders in Keras" post on "The Keras Blog"
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
        super(vae_model, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

        # Tracking various losses:
        self.total_loss_track = tf.keras.metrics.Mean(name = 'tot_loss')
        self.reconstruct_loss_track = tf.keras.metrics.Mean(name = 'reconstruct_loss')
        self.val_loss_track = tf.keras.metrics.Mean(name = 'val_loss')
        self.val_reconstruct_loss_track = tf.keras.metrics.Mean(name = 'reconstruct_val_loss')

    @property
    def metrics(self):
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
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruct_loss + kl_loss

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
        mu, log_var, z = self.encoder(inputs)
        reconstruct = self.decoder(z)
        return reconstruct

    def test_step(self, data):
        x, y = data
        mu, log_var, z = self.encoder(x)

        reconstruct = self.decoder(z)

        # Need to make sure that losses update:
        reconstruct_loss = tf.keras.losses.mse(x, reconstruct)

        kl_loss = (backend.exp(log_var) + backend.square(mu) - 1)
        kl_loss = backend.mean(backend.sum(kl_loss, axis=1))
        total_loss = reconstruct_loss + kl_loss

        self.val_loss_track.update_state(total_loss)
        self.val_reconstruct_loss_track.update_state(reconstruct_loss)

        # Return the losses with key so they'll show up in our verbose training routine
        return {
            'val_loss': total_loss,
            'reconstruct_val_loss': reconstruct_loss,
        }


def run_vae(label):

    # Load data:
    Xtrain, train_labels, Xval, val_labels = load_data()

    Ytrain, train_map = to_numeric(train_labels[label])
    Yval, val_map = to_numeric(val_labels[label])

    print(Ytrain)
    print(Yval)

    # First we set out hyperparameters:
    args = {
        'input_shape':(32, 32, 1),
        'bottleneck_size': 10,
        'reshape_size': (16, 16, 32),
        'batch_size': 1000
    }

    vae, encoder, decoder = build_model(args)

    '''
    # Need to define for loss
    inputs = tf.keras.layers.Input(shape = args['input_shape'], name = 'encode_input')
    outputs = decoder(encoder(inputs)[-1])

    print(vae.summary())
    print(encoder.summary())
    print(decoder.summary())

    print('layers', encoder.layers)

    reconstruction_loss = tf.keras.losses.mse(inputs, outputs)
    kl_loss = backend.exp(encoder.get_layer('log_var').output) + backend.square(encoder.get_layer('mu').output) - 1
    kl_loss = backend.sum(kl_loss, axis = -1)
    kl_loss *= 0.001 # Weighting term on KL divergence
    vae_loss = backend.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    '''

    vae = vae_model(encoder, decoder)

    # Training:
    vae.compile(optimizer = 'adam')
    history = vae.fit(Xtrain, epochs = 5, batch_size = args['batch_size'], validation_data = (Xval, Yval))
    
    # Plot the loss curve
    plt.plot(history.history['tot_loss'], label = 'Training')
    plt.plot(history.history['val_loss'], label = 'Validation')
    plt.title('Task 5 Training and Validation Loss (MSE + KL)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    args = {'input_shape':(32, 32, 1), 'bottleneck_size':5, 'reshape_size':(16, 16, 32) }
    #build_decoder(args)
    #build_encoder(args)
    run_vae('gender')


    