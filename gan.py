import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sort_data
import gen
import discriminator


#Initialize Weights
def weight_variables(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variables(shape):
    initial = tf.constant( 0.1, shape= shape)
    return tf.Variable(initial)

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


#Generator Parameters

z = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
#Z = tf.reshape(z, [-1, 150528])

label = tf.placeholder(tf.float32, shape=[None, 1])

G_W1 = tf.Variable(xavier_init([9217, 128]))
G_b1 = tf.Variable(tf.zeros(shape=[128]))

G_W2 = tf.Variable(xavier_init([128, 150528]))
G_b2 = tf.Variable(tf.zeros(shape=[150528]))

#Covolution 1
#conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
conv1_W = weight_variables([11, 11, 3, 96])
conv1_b = bias_variables([96])

#conv2
#conv(5, 5, 256, 1, 1, group=2, name='conv2')
conv2_W = weight_variables([5, 5, 48, 256])
conv2_b = bias_variables([256])

#conv3
#conv(3, 3, 384, 1, 1, name='conv3')
conv3_W = weight_variables([3, 3, 256, 384])
conv3_b = bias_variables([384])

#conv4
#conv(3, 3, 384, 1, 1, group=2, name='conv4')
conv4_W = weight_variables([3, 3, 192, 384])
conv4_b = bias_variables([384])

#conv5
#conv(3, 3, 256, 1, 1, group=2, name='conv5')
conv5_W = weight_variables([3, 3, 192, 256])
conv5_b = bias_variables([256])


theta_G = [G_W1, G_W2, G_b1, G_b2, conv1_W, conv1_b,
                                    conv2_W, conv2_b,
                                    conv3_W, conv3_b,
                                    conv4_W, conv4_b,
                                    conv5_W, conv5_b]

X_mb, Z_mb, labels = sort_data.Dataset()

'''
def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])
'''
#Discriminator Parameters

x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])

D_W1 = tf.Variable(xavier_init([9216, 1024]))
D_b1 = tf.Variable(tf.zeros(shape=[1024]))

D_W2 = tf.Variable(xavier_init([1024, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

#Covolution 1
#conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
Dconv1_W = weight_variables([11, 11, 3, 96])
Dconv1_b = bias_variables([96])

#conv2
#conv(5, 5, 256, 1, 1, group=2, name='conv2')
Dconv2_W = weight_variables([5, 5, 48, 256])
Dconv2_b = bias_variables([256])

#conv3
#conv(3, 3, 384, 1, 1, name='conv3')
Dconv3_W = weight_variables([3, 3, 256, 384])
Dconv3_b = bias_variables([384])

#conv4
#conv(3, 3, 384, 1, 1, group=2, name='conv4')
Dconv4_W = weight_variables([3, 3, 192, 384])
Dconv4_b = bias_variables([384])

#conv5
#conv(3, 3, 256, 1, 1, group=2, name='conv5')
Dconv5_W = weight_variables([3, 3, 192, 256])
Dconv5_b = bias_variables([256])

theta_D = [D_W1, D_W2, D_b1, D_b2, Dconv1_W, Dconv1_b,
                                    Dconv2_W, Dconv2_b,
                                    Dconv3_W, Dconv3_b,
                                    Dconv4_W, Dconv4_b,
                                    Dconv5_W, Dconv5_b]

'''
def discriminator(x):
    D_h1 = tf.nn.relu(tf.matmul(x, D_W1) + D_b1)
    D_logit = tf.matmul(D_h1, D_W2) + D_b2
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig
'''
print 'Original',z.shape
G_sample = gen.generator(z, theta_G, label)
D_real, D_logit_real = discriminator.discriminator(x, theta_D)
print 'Generated', G_sample.shape
G_sample = tf.reshape(G_sample, [-1, 224, 224, 3])
D_fake, D_logit_fake = discriminator.discriminator(G_sample, theta_D)

# D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
# G_loss = -tf.reduce_mean(tf.log(D_fake))

# Alternative losses:
# -------------------
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=D_logit_real, logits=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=D_logit_fake, logits=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=D_logit_fake, logits=tf.ones_like(D_logit_fake)))

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)



if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0

'''
if it % 1000 == 0:
    samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})
    
    fig = plot(samples)
    plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
    i += 1
    plt.close(fig)
'''
init = tf.initialize_all_variables()


with tf.Session() as sess:
    
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for it in range(1000000):
        mb_X = sess.run(X_mb)
        mb_Z = sess.run(Z_mb)
        mb_label = sess.run(labels)
        
        _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={x: mb_X, z: mb_Z, label: mb_label})
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={x: mb_X, z: mb_Z, label: mb_label})

        if it % 1 == 0:
            print('Iter: {}'.format(it))
            print('D loss: {:.4}'. format(D_loss_curr))
            print('G_loss: {:.4}'.format(G_loss_curr))
            print()