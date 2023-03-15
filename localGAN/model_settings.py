import tensorflow as tf
import keras

import os
import random
import numpy as np
import pathlib


SHAPE = [476,476,3]
RANDOM_SEED = 101
random.seed(RANDOM_SEED)
AUTOTUNE = tf.data.AUTOTUNE
latent_dim = 512


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

class GAN(keras.Model):
    def __init__(self, generator, discriminator, latent_dim):
        super(GAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim


    def compile(self, generator_optimizer, discriminator_optimizer, loss_fn):
        super(GAN, self).compile()
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.loss_fn = loss_fn
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")


    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]


    def train_step(self, real_images):     
        batch_size = tf.shape(real_images)[0]

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))     
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(random_latent_vectors)
            real_output = self.discriminator(real_images)
            fake_output = self.discriminator(generated_images)   
            real_loss = self.loss_fn(tf.ones_like(real_output), real_output)
            fake_loss = self.loss_fn(tf.zeros_like(fake_output), fake_output)
            gen_loss = self.loss_fn(tf.ones_like(fake_output), fake_output)
            disc_loss = real_loss + fake_loss
        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        # random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))    
        # with tf.GradientTape() as disc_tape:
        #     generated_images = self.generator(random_latent_vectors)
        #     real_output = self.discriminator(real_images)
        #     fake_output = self.discriminator(generated_images)     
        #     real_loss = self.loss_fn(tf.ones_like(real_output), real_output)
        #     fake_loss = self.loss_fn(tf.zeros_like(fake_output), fake_output)
        #     disc_loss = real_loss + fake_loss
        # gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
        # self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        self.g_loss_metric.update_state(gen_loss)
        self.d_loss_metric.update_state(disc_loss)

        return {"Gen Loss": gen_loss,  "Disc Loss": disc_loss}


    def test_step(self, test_images):
        predictions = self.discriminator(test_images, training=False)
        return {"discriminator": predictions}


    def predict_step(self, image):
        image = keras.layers.CenterCrop(SHAPE[0], SHAPE[1])(image)
        image = tf.image.per_image_standardization(image)
        image = tf.math.divide(tf.math.add(image,tf.math.abs(tf.math.reduce_min(image))),
                          tf.math.add(tf.math.reduce_max(image),tf.math.abs(tf.math.reduce_min(image)))+0.00001)
        return self.discriminator(image, training=False)


def Generator(SHAPE):
    init= tf.keras.initializers.RandomNormal(mean=0.0,stddev=0.02)
    inputs = keras.Input(shape=(latent_dim,))

    x = inputs
    x = keras.layers.Dense(15*15*48, use_bias=False)(x)
    x = keras.layers.BatchNormalization(momentum=0.5)(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)
    x = keras.layers.Reshape((15,15,48))(x)

    x = keras.layers.Conv2DTranspose(512, (3, 3), padding='same', use_bias=False, strides=(2,2), kernel_initializer=init)(x)
    x = keras.layers.BatchNormalization(momentum=0.5)(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)

    for filters in [256,192,174,142]:
        x = keras.layers.Conv2DTranspose(filters, (5, 5), padding='same', use_bias=False, strides=(2,2), kernel_initializer=init)(x)
        x = keras.layers.BatchNormalization(momentum=0.5)(x)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)

    x = keras.layers.Conv2D(3, (5, 5), padding='valid', kernel_initializer=init, use_bias=False)(x)
    output = keras.layers.BatchNormalization(momentum=0.5)(x)

    model = keras.models.Model(inputs, output, name='Generator')
    return model

def Discriminator(input_shape):
    inputs = keras.Input(shape=input_shape)

    x = inputs
    x = keras.layers.Conv2D(64, (3, 3), padding='same', input_shape=input_shape)(x)
    x = keras.layers.LeakyReLU()(x)

    for filters in [96,128,164,196,212]:
        x = keras.layers.Conv2D(filters, (5, 5), strides=(2,2), padding='same')(x)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Conv2D(256, (5, 5), strides=(2,2), padding='same')(x)
    x = keras.layers.LeakyReLU(alpha=0.2)(x)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    output = keras.layers.Dense(1, activation='sigmoid')(x)

    model = keras.models.Model(inputs, output, name='Discriminator')
    return model