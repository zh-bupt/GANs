import os
import sys
sys.path.append(r'/home/wxr/zhsworkingspace/code')
from common.ops import *
from tensorflow.examples.tutorials.mnist import input_data
import time

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
iter = 80000
batch_size = 128
lr = 0.0002
beta1=0.5

mnist = input_data.read_data_sets('../MNIST/', one_hot=True)
train = mnist.train


def generator(z, y, training=True):
    h1_z = fully_connect(z, 200, name='g_h1_z_fc')
    h1_z = tf.nn.relu(tf.layers.batch_normalization(h1_z, training=training, name='g_h1_z_bn'))
    h1_y = fully_connect(y, 1000, name='g_h1_y_fc')
    h1_y = tf.nn.relu(tf.layers.batch_normalization(h1_y, training=training, name='g_h1_y_bn'))
    h1 = tf.concat([h1_z, h1_y], axis=1)

    h2 = tf.nn.dropout(h1, keep_prob=0.5)
    h2 = fully_connect(h2, 784, name='g_h2_fc')
    return tf.nn.sigmoid(h2)

def discriminator(images, y, reuse=False, training=True):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    h1_img = maxout(images, 240, name='d_h1_img_maxout')
    h1_y = maxout(y, 50, name='d_h1_y_maxout')
    h1 = tf.concat([h1_img, h1_y], axis=1)
    h1 = tf.nn.dropout(h1, keep_prob=0.5)

    h2 = maxout(h1, 240, name='d_h2_maxout', pieces=4)
    h2 = tf.nn.dropout(h2, keep_prob=0.5)

    h3 = fully_connect(h2, 1, name='d_h3_fc')
    return tf.nn.sigmoid(h3), h3

def sampler(z, y, training=False):
    tf.get_variable_scope().reuse_variables()
    return generator(z, y, training=training)


global_step = tf.Variable(0, name='global_step', trainable=False)
increment_op = tf.assign_add(global_step, tf.constant(1))

z = tf.placeholder(tf.float32, [None, 100], name='z')
y = tf.placeholder(tf.float32, [None, 10], name='y')
images = tf.placeholder(tf.float32, [None, 784], name='images')
sample_y = sample_labels()


with tf.variable_scope(tf.get_variable_scope()) as scope:
    G = generator(z, y)
    D, D_logits = discriminator(images, y)
    D_, D_logits_ = discriminator(G, y, reuse=True)
    samples = sampler(z, y)

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=D_logits, labels=tf.ones_like(D_logits)
))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=D_logits_, labels=tf.zeros_like(D_logits_)
))
d_loss = d_loss_real + d_loss_fake
tf.summary.scalar('discriminator loss', d_loss)

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=D_logits_, labels=tf.ones_like(D_logits_)
))
tf.summary.scalar('generator loss', g_loss)
merge_summary = tf.summary.merge_all()

vars = tf.trainable_variables()
d_vars = [var for var in vars if 'd_' in var.name]
g_vars = [var for var in vars if 'g_' in var.name]

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    d_optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).\
        minimize(d_loss, var_list=d_vars, global_step=global_step)
    g_optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).\
        minimize(g_loss, var_list=g_vars, global_step=global_step)

samples_path = './out'+ time.strftime('_%Y-%m-%d_%H:%M:%S', time.localtime()) +'/'
if not os.path.exists(samples_path):
    os.makedirs(samples_path)

with tf.Session(config=config) as sess:
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter('./summary', sess.graph)
    sess.run(tf.global_variables_initializer())
    for i in range(iter):
        batch = mnist.train.next_batch(batch_size)
        batch_images = np.array(batch[0])
        batch_labels = batch[1]
        batch_z = np.random.uniform(-1, 1, size=(batch_size, 100))
        sess.run(d_optimizer, feed_dict={
            images: batch_images, y: batch_labels, z: batch_z
        })
        sess.run(g_optimizer, feed_dict={
            images: batch_images, y: batch_labels, z: batch_z
        })

        train_summary = sess.run(
            merge_summary, feed_dict={images: batch_images, y: batch_labels, z: batch_z}
        )
        summary_writer.add_summary(train_summary, i)
        if i % 1000 == 0:
            d_err = d_loss.eval(feed_dict={
                images: batch_images, y: batch_labels, z: batch_z
            })
            g_err = g_loss.eval(feed_dict={
                z: batch_z, y: batch_labels
            })
            print('i:[{}], d_loss: {}, g_loss: {}'.format(i, d_err, g_err))
            batch_z_sample = np.random.uniform(-1, 1, size=(100, 100))
            sample = sess.run(samples, feed_dict={z: batch_z_sample, y: sample_y})
            save_images(
                np.reshape(sample, [-1, 28, 28, 1]),
                [10, 10],
                samples_path + 'i_{}.png'.format(str(i).zfill(4))
            )
            print('save image')

        if i == 1000:
            saver.save(sess, './checkpoint_dir/cgan')
            print()
