import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
import numpy as np
import scipy


def weight_variable(shape, name, initializer=xavier_initializer(), trainable=True):
    return tf.get_variable(name=name, shape=shape, trainable=trainable,
                    initializer=initializer)

def bias_variable(shape, name, trainable=True):
    return tf.get_variable(name=name, shape=shape, trainable=trainable,
                    initializer=tf.constant_initializer(0, dtype=tf.float32))

def conv2d(input, output_channels, name, ksize=[5, 5], strides=[2, 2]):
    input_shape = input.get_shape().as_list()
    with tf.variable_scope(name):
        kernels = weight_variable(
            [ksize[0], ksize[1], input_shape[-1], output_channels], name='weights'
        )
        bias = bias_variable([output_channels], name='biases')
        conv = tf.nn.conv2d(input, kernels, strides=[1, strides[0], strides[1], 1], padding='SAME') + bias
        return conv

def deconv2d(input, output_shape, name, ksize=[5, 5], strides=[2, 2]):
    input_shape = input.get_shape().as_list()
    output_shape = [tf.shape(input)[0]] + output_shape
    with tf.variable_scope(name):
        kernels = weight_variable(
            [ksize[0], ksize[1], output_shape[-1], input_shape[-1]], name='weights'
        )
        bias = bias_variable([output_shape[-1]], name='biases')
        deconv = tf.nn.conv2d_transpose(
            input, kernels, output_shape, strides=[1, strides[0], strides[1], 1], padding='SAME'
        ) + bias
        return deconv

def fully_connect(input, channels_out, name):
    channels_in = input.get_shape().as_list()[-1]
    with tf.variable_scope(name):
        w = weight_variable([channels_in, channels_out], name='weights')
        b = bias_variable([channels_out], name='biases')
        return tf.matmul(input, w) + b

def maxout(input, channels_out, name, pieces=5):
    channels_in = input.get_shape().as_list()[-1]
    with tf.variable_scope(name):
        w = weight_variable([channels_in, channels_out, pieces], name='weights')
        b = bias_variable([channels_out, pieces], name='bias')
        return tf.reduce_max(tf.tensordot(input, w, axes=1) + b, axis=-1)

def relu(value):
    return tf.nn.relu(value)

def leaky_relu(value, leak=0.2):
    return tf.maximum(value, value * leak)

def conv_cond_concat(value, cond):
    value_shape = value.get_shape().as_list()
    cond_shape = cond.get_shape().as_list()
    return tf.concat(
        [value, cond * tf.ones(value_shape[1:3] + cond_shape[3:])], axis=3
    )

def sample_labels():
    labels = np.zeros((100, 10))
    for i in range(10):
        labels[i * 10:(i + 1) * 10, i] = 1.
    return labels

def save_images(images, size, path):
    # img = (images + 1.0) / 2.0
    h, w = images.shape[1], images.shape[2]

    merge_img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        if j >= size[0]:
            break
        merge_img[j * h:j * h + h, i * w:i * w + w, :] = image
    return scipy.misc.imsave(path, merge_img)
