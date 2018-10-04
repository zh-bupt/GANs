from tensorflow.examples.tutorials.mnist import input_data
from common.ops import *


def generator(z, labels, training=True):
    yb = tf.reshape(labels, [-1, 1, 1, 10])

    z = tf.concat([z, labels], axis=1)
    h1 = fully_connect(z, 32, name='g_h1_fc')
    h1 = tf.nn.relu(tf.layers.batch_normalization(h1, training=training, name='g_h1_bn'))

    h1 = tf.concat([h1, labels], axis=1)
    h2 = fully_connect(h1, 49 * 64, name='g_h2_fc')
    h2 = tf.nn.relu(tf.layers.batch_normalization(h2, training=training, name='g_h2_bn'))
    h2 = tf.reshape(h2, [-1, 7, 7, 64])

    h2 = conv_cond_concat(h2, yb)
    h3 = deconv2d(h2, [14, 14, 64], name='g_h3_deconv')
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
    h1 = conv2d(x, 5, name='d_h1_conv')
    h1 = leaky_relu(tf.layers.batch_normalization(h1, training=training, name='d_h1_bn'))

    h1 = conv_cond_concat(h1, yb)
    h2 = conv2d(h1, 10, name='d_h2_conv')
    h2 = leaky_relu(tf.layers.batch_normalization(h2, training=training, name='d_h2_bn'))

    # h2_shape = h2.get_shape().as_list()
    # h2 = tf.reshape(h2, [-1, np.prod(h2_shape[1:])])
    h2 = tf.layers.flatten(h2)
    h2 = tf.concat([h2, lables], axis=1)
    h3 = fully_connect(h2, 128, name='d_h3_fc')
    h3 = leaky_relu(tf.layers.batch_normalization(h3, training=training, name='d_h3_bn'))

    h3 = tf.concat([h3, lables], axis=1)
    h4 = fully_connect(h3, 1, name='d_h4_fc')

    return tf.nn.sigmoid(h4), h4

def sampler(z, y, training=False):
    tf.get_variable_scope().reuse_variables()
    return generator(z, y, training=training)


epochs = 25
batch_size = 128
sample_size = 10

mnist = input_data.read_data_sets('../MNIST/', one_hot=True)
train = mnist.train

global_step = tf.Variable(0, name='global_step', trainable=False)
y = tf.placeholder(tf.float32, [None, 10], name='y')
images = tf.placeholder(tf.float32, [None, 28, 28, 1], name='real_images')
z = tf.placeholder(tf.float32, [None, 100], name='z')
# z_sample = tf.placeholder(tf.float32, [None, 100], name='z_sample')

sample_labels = sample_labels()

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

vars = tf.trainable_variables()
d_vars = [var for var in vars if 'd_' in var.name]
g_vars = [var for var in vars if 'g_' in var.name]

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    d_optimizer = tf.train.AdamOptimizer().\
        minimize(d_loss, var_list=d_vars, global_step=global_step)
    g_optimizer = tf.train.AdamOptimizer().\
        minimize(g_loss, var_list=g_vars, global_step=global_step)

# TODO: add summary
with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
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
            if i % 10 == 0:
                d_err = d_loss.eval(feed_dict={
                    images: batch_images, y: batch_labels, z: batch_z
                })
                g_err = g_loss.eval(feed_dict={
                    z: batch_z, y: batch_labels
                })
                print('epoch:[{}], i:[{}], d_loss: {}, g_loss: {}'.format(epoch, i, d_err, g_err))
                batch_z_sample = np.random.uniform(-1, 1, size=(100, 100))
                sample = sess.run(samples, feed_dict={z: batch_z_sample, y: sample_labels})
                samples_path = './out/'
                save_images(sample, [10, 10], samples_path + 'epoch_%d_i_%d.png' % (epoch, i))
                print('save image')

            if i == int(55000 / batch_size) - 1:
                saver.save(sess, './checkpoint_dir/dcgan')
                print('model saved')
