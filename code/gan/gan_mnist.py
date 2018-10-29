import sys
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sys.path.append('../')
from common.ops import *


config = tf.ConfigProto()
config.gpu_options.allow_growth = True

epochs = 60000
batch_size = 128
Z_dim = 100

mnist = input_data.read_data_sets('../MNIST/', one_hot=True)


def generator(z, training=True):
    h1 = fully_connect(z, 1200, name='g_h1_fc')
    h1 = tf.nn.relu(
        tf.layers.batch_normalization(h1, training=training, name='g_h1_bn'))

    h2 = fully_connect(h1, 1200, name='g_h2_fc')
    h2 = tf.nn.relu(
        tf.layers.batch_normalization(h2, training=training, name='g_h2_bn'))

    h3 = fully_connect(h2, 784, name='g_h3_fc')
    h3 = tf.nn.sigmoid(h3)
    return h3


def discriminator(images, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    h1 = maxout(images, 240, name='d_h1_maxout')
    h1 = tf.nn.dropout(h1, 0.8)
    h2 = maxout(h1, 240, name='d_h2_maxout')
    h2 = tf.nn.dropout(h2, 0.5)
    h3 = fully_connect(h2, 1, name='d_h3_fc')
    return tf.nn.sigmoid(h3), h3


def sampler(z, training=False):
    tf.get_variable_scope().reuse_variables()
    return generator(z, training=training)


X = tf.placeholder(tf.float32, shape=[None, 784], name='X')
Z = tf.placeholder(tf.float32, shape=[None, 100], name='Z')

global_step = tf.Variable(0, trainable=False, name='global_step')
increment_op = tf.assign_add(global_step, tf.constant(1))

with tf.variable_scope(tf.get_variable_scope()) as scope:
    G = generator(Z)
    D_real, D_real_logit = discriminator(X)
    D_fake, D_fake_logit = discriminator(G, reuse=True)
    sample = sampler(Z)

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=D_real_logit,
    labels=0.9 * tf.ones_like(D_real_logit)
))

D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=D_fake_logit,
    labels=tf.zeros_like(D_fake_logit)
))
D_loss = D_loss_real + D_loss_fake
tf.summary.scalar('discriminator loss', D_loss)

# 注意此处使用了单侧标签平滑
# 即使用一个目标值1−α为真样本， 并且使用目标值0+β为伪样本
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=D_fake_logit,
    labels=0.9 * tf.ones_like(D_fake_logit)
))
tf.summary.scalar('generator loss', G_loss)
merge_summary = tf.summary.merge_all()

vars = tf.trainable_variables()
theta_d = [var for var in vars if 'd_' in var.name]
theta_g = [var for var in vars if 'g_' in var.name]
D_optimizer = tf.train.AdamOptimizer().\
    minimize(D_loss, var_list=theta_d, global_step=global_step)
G_optimizer = tf.train.AdamOptimizer().\
    minimize(G_loss, var_list=theta_g, global_step=global_step)

sample_path = './out_mnist/'
if not os.path.exists(sample_path):
    os.makedirs(sample_path)

Z_sample_batch = np.random.uniform(-1., 1., size=[25, Z_dim])

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter('./summary/', sess.graph)
    for i in range(epochs):
        if i % 1000 == 0:
            samples = sess.run(sample, feed_dict={Z: Z_sample_batch})
            save_images(
                np.reshape(samples, [-1, 28, 28, 1]),
                [5, 5],
                sample_path + '{}.png'.format(str(i).zfill(5))
            )

        X_batch, _ = mnist.train.next_batch(batch_size)
        Z_batch = np.random.uniform(-1., 1., size=[batch_size, Z_dim])

        sess.run(D_optimizer, feed_dict={X: X_batch, Z: Z_batch})
        sess.run(G_optimizer, feed_dict={Z: Z_batch})

        train_summary = sess.run(
            merge_summary, feed_dict={X: X_batch, Z: Z_batch}
        )
        summary_writer.add_summary(train_summary, i)

        if i % 1000 == 0:
            D_cur_loss, G_cur_loss = sess.run(
                [D_loss, G_loss], feed_dict={X: X_batch, Z: Z_batch})
            print('Iter: {}'.format(i))
            print('D loss: {}'.format(D_cur_loss))
            print('G loss: {}'.format(G_cur_loss))
            print()
            saver.save(sess, './checkpoint_dir/gan')
