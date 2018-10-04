import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tensorflow.examples.tutorials.mnist import input_data
import os


sess = tf.InteractiveSession()


epochs = 50000
batch_size = 128
Z_dim = 100

mnist = input_data.read_data_sets('../MNIST/', one_hot=True)

def get_weights(shape, name):
    return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

def get_bias(shape, name):
    return tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer)

# discriminator
X = tf.placeholder(tf.float32, shape=[None, 784], name='X')

D_W1 = get_weights([784, 128], 'D_W1')
D_b1 = get_bias([128], 'D_b1')

D_W2 = get_weights([128, 1], 'D_W2')
D_b2 = get_bias([1], 'D_b2')

theta_D = [D_W1, D_b1, D_W2, D_b2]

# generator
Z = tf.placeholder(tf.float32, shape=[None, 100], name='Z')

G_W1 = get_weights([100, 128], 'G_W1')
G_b1 = get_bias([128], 'G_b1')

G_W2 = get_weights([128, 784], 'G_W2')
G_b2 = get_bias([784], 'G_b2')

theta_G = [G_W1, G_b1, G_W2, G_b2]

def generator(z):
    G_A1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
    G_A2 = tf.nn.sigmoid(tf.matmul(G_A1, G_W2) + G_b2)
    return G_A2

def discriminator(x):
    D_A1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_A1, D_W2) + D_b2
    D_A2 = tf.nn.sigmoid(D_logit)
    return D_A2, D_logit

G_sample = generator(Z)
D_real, D_real_logit = discriminator(X)
D_fake, D_fake_logit = discriminator(G_sample)

# D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
# G_loss = -tf.reduce_mean(tf.log(D_fake))

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=D_real_logit,
    labels=0.9 * tf.ones_like(D_real_logit)
))

D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=D_fake_logit,
    labels=tf.zeros_like(D_fake_logit)
))
D_loss = D_loss_real + D_loss_fake

# 注意此处使用了单侧标签平滑
# 即使用一个目标值1−α为真样本， 并且使用目标值0+β为伪样本
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=D_fake_logit,
    labels=0.9 * tf.ones_like(D_fake_logit)
))
D_optimizer = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_optimizer = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

def plot_result(samples):
    fig = plt.figure(figsize=(4, 4))
    grid = gridspec.GridSpec(4, 4)
    grid.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(grid[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

sess.run(tf.global_variables_initializer())

if __name__ == '__main__':
    if not os.path.exists('./out/'):
        os.makedirs('./out/')
    num = 0
    saver = tf.train.Saver()
    for i in range(epochs):
        if i % 1000 == 0:
            samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})
            fig = plot_result(samples)
            plt.savefig('./out/{}.png'.format(str(num).zfill(3)), bbox_inches='tight')
            num = num + 1
            plt.close(fig)

        X_batch, _ = mnist.train.next_batch(batch_size)

        _, D_cur_loss = sess.run([D_optimizer, D_loss],
                 feed_dict={X: X_batch, Z: sample_Z(batch_size, Z_dim)})
        _, G_cur_loss = sess.run([G_optimizer, G_loss],
                 feed_dict={Z: sample_Z(batch_size, Z_dim)})

        if i % 1000 == 0:
            print('Iter: {}'.format(i))
            print('D loss: {}'.format(D_cur_loss))
            print('G loss: {}'.format(G_cur_loss))
            print()
            saver.save(sess, './checkpoint_dir/gan')
