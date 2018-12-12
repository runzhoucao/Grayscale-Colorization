import numpy as np
import tensorflow as tf
from .ops import conv2d, conv2d_transpose, pixelwise_accuracy, to_lab


class Discriminator(object):
    def __init__(self, name, kernels):
        self.name = name
        self.kernels = kernels
        self.var_list = []

    def create(self, inputs, kernel_size=None, seed=None, reuse_variables=None):
        output = inputs
        with tf.variable_scope(self.name, reuse=reuse_variables):
            for index, kernel in enumerate(self.kernels):

                # not use batch-norm in the first layer
                bnorm = False if index == 0 else True
                name = 'conv' + str(index)
                output = conv2d(
                    inputs=output,
                    name=name,
                    kernel_size=kernel_size,
                    filters=kernel[0],
                    strides=kernel[1],
                    bnorm=bnorm,
                    activation=tf.nn.leaky_relu,
                    seed=seed
                )

                if kernel[2] > 0:
                    output = tf.nn.dropout(output, keep_prob=1 - kernel[2], name='dropout_' + name, seed=seed)

            output = conv2d(
                inputs=output,
                name='conv_last',
                filters=1,
                kernel_size=4,                  # last layer kernel size = 4
                strides=1,                      # last layer stride = 1
                bnorm=False,                    # do not use batch-norm for the last layer
                seed=seed
            )

            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

            return output


class Generator(object):
    def __init__(self, name, encoder_kernels, decoder_kernels, output_channels=3, img_size=64):
        self.name = name
        self.encoder_kernels = encoder_kernels
        self.decoder_kernels = decoder_kernels
        self.output_channels = output_channels
        self.var_list = []
        self.img_size = img_size

    def create(self, inputs, kernel_size=None, seed=None, reuse_variables=None):
        output = inputs
        print(tf.reduce_max(inputs))
        print(tf.reduce_min(inputs))

        with tf.variable_scope(self.name, reuse=reuse_variables):

            layers = []

            # encoder branch
            for index, kernel in enumerate(self.encoder_kernels):

                name = 'conv' + str(index)
                output = conv2d(
                    inputs=output,
                    name=name,
                    kernel_size=kernel_size,
                    filters=kernel[0],
                    strides=kernel[1],
                    activation=tf.nn.leaky_relu,
                    seed=seed
                )

                # save contracting path layers to be used for skip connections
                layers.append(output)

                if kernel[2] > 0:
                    output = tf.nn.dropout(output, keep_prob=1 - kernel[2], name='dropout_' + name, seed=seed)

            # decoder branch
            size = self.img_size
            for index, kernel in enumerate(self.decoder_kernels):

                name = 'deconv' + str(index)
                output = conv2d_transpose(
                    inputs=output,
                    name=name,
                    kernel_size=kernel_size,
                    filters=kernel[0],
                    strides=kernel[1],
                    activation=tf.nn.relu,
                    seed=seed
                )

                if kernel[2] > 0:
                    output = tf.nn.dropout(output, keep_prob=1 - kernel[2], name='dropout_' + name, seed=seed)

                if index == 1 or index == 3:
                    # upsampling
                    output = tf.image.resize_images(output, [size, size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
                    size = size * 2

                # concat the layer from the contracting path with the output of the current layer
                # concat only the channels (axis=3)
                # output = tf.concat([layers[len(layers) - index - 2], output], axis=3)

            # output 256 256 2

            # output layer
            output = conv2d(
                inputs=output,
                name='output_layer',
                filters=2,  # number of output chanels
                kernel_size=1,  # last layer kernel size = 1
                strides=1,  # last layer stride = 1
                bnorm=False,  # do not use batch-norm for the last layer
                activation=tf.nn.tanh,  # tanh activation function for the output
                seed=seed
            )

            output = tf.image.resize_images(output, [size, size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            output = to_lab(output, inputs)  # return LAB space image

            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

            return output
