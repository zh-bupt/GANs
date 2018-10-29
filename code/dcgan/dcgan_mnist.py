from tensorflow.examples.tutorials.mnist import input_data
import sys
sys.path.append('../')
from common.ops import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def generator(z, labels, training=True):
    yb = tf.reshape(labels, [-1, 1, 1, 10])

    z = tf.concat([z, labels], axis=1)
    h1 = fully_connect(z, 1024, name='g_h1_fc')
    h1 = tf.nn.relu(tf.layers.batch_normalization(h1, training=training, name='g_h1_bn'))

    h1 = tf.concat([h1, labels], axis=1)
    h2 = fully_connect(h1, 128 * 49, name='g_h2_fc')
    h2 = tf.nn.relu(tf.layers.batch_normalization(h2, training=training, name='g_h2_bn'))
    h2 = tf.reshape(h2, [-1, 7, 7, 128])

    h2 = conv_cond_concat(h2, yb)
    h3 = deconv2d(h2, [14, 14, 128], name='g_h3_deconv')
    h3 = tf.nn.relu(tf.layers.batch_normalization(h3, training=training, name='g_h3_bn'))

    h3 = conv_cond_concat(h3, yb)
    h4 = deconv2d(h3, [28, 28, 1], name='g_h4_deconv')
    h4 = tf.nn.sigmoid(h4)

    return h4

def discriminator(images, lables, reuse=False, training=True):
    if reuse:
        tf.get_variable_scope().reuse_variables()

    yb = tf.reshape(lables, [-1, 1, 1, 10])

    x = conv_cond_concat(images, yb)
    h1 = conv2d(x, 11, name='d_h1_conv')
    h1 = leaky_relu(tf.layers.batch_normalization(h1, training=training, name='d_h1_bn'))

    h1 = conv_cond_concat(h1, yb)
    h2 = conv2d(h1, 74, name='d_h2_conv')
    h2 = leaky_relu(tf.layers.batch_normalization(h2, training=training, name='d_h2_bn'))

    # h2_shape = h2.get_shape().as_list()
    # h2 = tf.reshape(h2, [-1, np.prod(h2_shape[1:])])
    h2 = tf.layers.flatten(h2)
    h2 = tf.concat([h2, lables], axis=1)
    h3 = fully_connect(h2, 1024, name='d_h3_fc')
    h3 = leaky_relu(tf.layers.batch_normalization(h3, training=training, name='d_h3_bn'))

    h3 = tf.concat([h3, lables], axis=1)
    h4 = fully_connect(h3, 1, name='d_h4_fc')

    return tf.nn.sigmoid(h4), h4

def sampler(z, y, training=False):
    tf.get_variable_scope().reuse_variables()
    return generator(z, y, training=training)


epochs = 20
batch_size = 128
lr = 0.0002
beta1 = 0.5

mnist = input_data.read_data_sets('../MNIST/', one_hot=True)
train = mnist.train

global_step = tf.Variable(0, name='global_step', trainable=False)
increment_op = tf.assign_add(global_step, tf.constant(1))
y = tf.placeholder(tf.float32, [None, 10], name='y')
images = tf.placeholder(tf.float32, [None, 28, 28, 1], name='real_images')
z = tf.placeholder(tf.float32, [None, 100], name='z')

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
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=D_logits_, labels=tf.ones_like(D_logits_)
))
tf.summary.scalar('discriminator loss', d_loss)
tf.summary.scalar('generator loss', g_loss)
merged_summary = tf.summary.merge_all()

vars = tf.trainable_variables()
d_vars = [var for var in vars if 'd_' in var.name]
g_vars = [var for var in vars if 'g_' in var.name]

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    d_optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1).\
        minimize(d_loss, var_list=d_vars, global_step=global_step)
    g_optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1).\
        minimize(g_loss, var_list=g_vars, global_step=global_step)

batch_sample_y = sample_labels()
sample_path = './out_mnist/'
if not os.path.exists(sample_path):
    os.makedirs(sample_path)


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter('./summary_mnist/', sess.graph)
    for epoch in range(epochs):
        for i in range(int(55000 / batch_size)):
            batch = mnist.train.next_batch(batch_size)
            batch_images = np.array(batch[0]).reshape((-1, 28, 28, 1))
            batch_labels = batch[1]
            batch_z = np.random.uniform(-1, 1, size=(batch_size, 100))
            sess.run(d_optimizer, feed_dict={
                images: batch_images, y: batch_labels, z: batch_z
            })
            sess.run(g_optimizer, feed_dict={
                images: batch_images, y: batch_labels, z: batch_z
            })
            sess.run(g_optimizer, feed_dict={
                images: batch_images, y: batch_labels, z: batch_z
            })

            train_summary = sess.run(merged_summary, feed_dict={
                images: batch_images, z: batch_z, y: batch_labels
            })
            summary_writer.add_summary(train_summary, global_step=global_step.eval())
            if i % 100 == 0:
                d_err = d_loss.eval(feed_dict={
                    images: batch_images, y: batch_labels, z: batch_z
                })
                g_err = g_loss.eval(feed_dict={
                    z: batch_z, y: batch_labels
                })
                print('epoch:[{}], i:[{}], d_loss: {}, g_loss: {}'.format(epoch, i, d_err, g_err))

            if i == int(55000 / batch_size) - 1:
                saver.save(sess, './checkpoint_dir_mnist/dcgan_mnist')
                print('model saved')

                # sample
                batch_sample_z = np.random.uniform(-1, 1, size=(100, 100))
                sample_images = sess.run(samples, feed_dict={
                    z: batch_sample_z, y: batch_sample_y
                })
                save_images(
                    np.reshape(sample_images, [-1, 28, 28, 1]),
                    [10, 10],
                    sample_path + 'epoch_{}_i_{}.png'.format(str(epoch).zfill(2), str(i).zfill(3))
                )
