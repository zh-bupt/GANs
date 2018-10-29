import sys
sys.path.append('../')
from common.ops import *
from common import cifar

import tensorflow as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def generator(z, reuse=False, training=True):
    with tf.variable_scope('generator', reuse=reuse):
        # 1 * 1 * 100 -> 4 * 4 * 512
        h0 = tf.layers.conv2d_transpose(z, 512, [4, 4], strides=(1, 1), padding='valid', name='g_h0_dconv')
        h0 = leaky_relu(tf.layers.batch_normalization(h0, training=training, name='g_h0_bn'))

        # 4 * 4 * 512 -> 8 * 8 * 256
        h1 = tf.layers.conv2d_transpose(h0, 256, [4, 4], strides=(2, 2), padding='same', name='g_h1_dconv')
        h1 = leaky_relu(tf.layers.batch_normalization(h1, training=training, name='g_h1_bn'))

        # 8 * 8 * 256 -> 16 * 16 * 128
        h2 = tf.layers.conv2d_transpose(h1, 128, [4, 4], strides=(2, 2), padding='same', name='g_h2_dconv')
        h2 = leaky_relu(tf.layers.batch_normalization(h2, training=training, name='g_h2_bn'))

        # 16 * 16 * 128 -> 32 * 32 * 3
        h3 = tf.layers.conv2d_transpose(h2, 3, [4, 4], strides=(2, 2), padding='same', name='g_h3_dconv')
        h3 = tf.nn.tanh(h3)

        return h3


def discriminator(images, reuse=False, training=True):
    with tf.variable_scope('discriminator', reuse=reuse):
        # 32 * 32 * 3 -> 16 * 16 * 128
        h0 = tf.layers.conv2d(images, 128, [4, 4], strides=(2, 2), padding='same', name='d_h0_conv')
        h0 = leaky_relu(tf.layers.batch_normalization(h0, training=training, name='d_h0_bn'))

        # 16 * 16 * 128 -> 8 * 8 * 256
        h1 = tf.layers.conv2d(h0, 256, [4, 4], strides=(2, 2), padding='same', name='d_h1_conv')
        h1 = leaky_relu(tf.layers.batch_normalization(h1, training=training, name='d_h1_bn'))

        # 8 * 8 * 256 -> 4 * 4 * 512
        h2 = tf.layers.conv2d(h1, 512, [4, 4], strides=(2, 2), padding='same', name='d_h2_conv')
        h2 = leaky_relu(tf.layers.batch_normalization(h2, training=training, name='d_h2_bn'))

        # 4 * 4 * 512 -> 1 * 1 * 1
        h3 = tf.layers.conv2d(h2, 1, [4, 4], strides=(1, 1), padding='valid', name='d_h3_conv')

        return tf.nn.sigmoid(h3), h3


def sampler(z):
    return generator(z, reuse=True, training=False)


lr = 0.0002
beta1 = 0.5
batch_size = 128
iter = 50000
Z_dim = 100

data_reader = cifar.Cifar10DataReader('../cifar-10/cifar-10-batches-py')

z = tf.placeholder(tf.float32, shape=(None, 1, 1, 100))
images = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))

G = generator(z)
D_real, D_real_logits = discriminator(images)
D_fake, D_fake_logits = discriminator(G, reuse=True)
sample = sampler(z)

D_loss_real = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real_logits)))
D_loss_fake = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake_logits)))
D_loss = D_loss_fake + D_loss_real
G_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake_logits)))
tf.summary.scalar('discriminator loss', D_loss)
tf.summary.scalar('generator loss', G_loss)
merge_summary = tf.summary.merge_all()


vars = tf.trainable_variables()
d_vars = [var for var in vars if 'd_' in var.name]
g_vars = [var for var in vars if 'g_' in var.name]

with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    d_optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1).\
        minimize(D_loss, var_list=d_vars)
    g_optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1).\
        minimize(G_loss, var_list=g_vars)

sample_path = './out_cifar10/'
if not os.path.exists(sample_path):
    os.makedirs(sample_path)
z_sample_batch = np.random.uniform(-1., 1., size=[25, 1, 1, Z_dim])


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter('./summary_cifar10/', sess.graph)
    for i in range(iter):
        if i % 1000 == 0:
            samples = sess.run(sample, feed_dict={z: z_sample_batch})
            save_images(
                (np.reshape(samples, [-1, 32, 32, 3]) + 1.) * 128.,
                [5, 5],
                sample_path + '{}.png'.format(str(i).zfill(5))
            )

        x_batch = data_reader.next_batch() / 128. - 1.
        z_batch = np.random.uniform(-1., 1., size=[batch_size, 1, 1, Z_dim])

        sess.run(d_optimizer, feed_dict={z: z_batch, images: x_batch})
        sess.run(g_optimizer, feed_dict={z: z_batch, images: x_batch})

        train_summary = sess.run(
            merge_summary, feed_dict={images: x_batch, z: z_batch}
        )
        summary_writer.add_summary(train_summary, i)

        if i % 100 == 0:
            D_cur_loss, G_cur_loss = sess.run(
                [D_loss, G_loss], feed_dict={images: x_batch, z: z_batch})
            print('Iter: {}'.format(i))
            print('D loss: {}'.format(D_cur_loss))
            print('G loss: {}'.format(G_cur_loss))
            print()
            saver.save(sess, './checkpoint_dir_cifar10/dcgan_cifar10/')
