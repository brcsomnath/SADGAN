import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sort_data


#Convolution

def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w,  padding="VALID", group=1):    
    #From https://github.com/ethereon/caffe-tensorflow
    
    c_i = input.get_shape()[-1]
    assert c_i%group==0
    assert c_o%group==0
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    
    
    if group==1:
        conv = convolve(input, kernel)
    else:
        input_groups = tf.split(input, group, 3)
        kernel_groups = tf.split(kernel, group, 3)
        output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
        conv = tf.concat(output_groups, 3)
    return  conv

#Max-pooling 
def max_pool(x, k_h, k_w, s_h, s_w, padding):
  return tf.nn.max_pool(x, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding= padding)

# Local response normalization
def lrn(x, radius, alpha, beta, bias):
    return tf.nn.local_response_normalization( x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)



'''
def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])
'''

def generator(z, theta_G, label):

    print 'G_sample',z.shape
    

    G_W1 = theta_G[0]
    G_b1 = theta_G[2]

    G_W2 = theta_G[1]
    G_b2 = theta_G[3]

    conv1_W = theta_G[4]
    conv1_b = theta_G[5]
    conv2_W = theta_G[6]
    conv2_b = theta_G[7]
    conv3_W = theta_G[8]
    conv3_b = theta_G[9]
    conv4_W = theta_G[10]
    conv4_b = theta_G[11]
    conv5_W = theta_G[12]
    conv5_b = theta_G[13]



    k_h = 11; k_w = 11; c_o = 96; s_h = 4; s_w = 4
    conv1_in = conv(z, conv1_W, conv1_b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
    conv1 = tf.nn.relu(conv1_in + conv1_b)

    #lrn1
    #lrn(2, 2e-05, 0.75, name='norm1')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn1 = lrn(conv1, radius, alpha, beta, bias)

    #maxpool1
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool1 = max_pool(lrn1, k_h, k_w, s_h, s_w, padding)

    
    k_h = 5; k_w = 5; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv2_in = conv(maxpool1, conv2_W, conv2_b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv2 = tf.nn.relu(conv2_in + conv2_b)

    #lrn2
    #lrn(2, 2e-05, 0.75, name='norm2')
    radius = 2; alpha = 2e-05; beta = 0.75; bias = 1.0
    lrn2 = lrn(conv2, radius, alpha, beta, bias)

    #maxpool2
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool2')                                                  
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool2 = max_pool(lrn2, k_h, k_w, s_h, s_w,padding)


    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 1
    conv3_in = conv(maxpool2, conv3_W, conv3_b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv3 = tf.nn.relu(conv3_in + conv3_b)

    
    k_h = 3; k_w = 3; c_o = 384; s_h = 1; s_w = 1; group = 2
    conv4_in = conv(conv3, conv4_W, conv4_b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv4 = tf.nn.relu(conv4_in + conv4_b)


    k_h = 3; k_w = 3; c_o = 256; s_h = 1; s_w = 1; group = 2
    conv5_in = conv(conv4, conv5_W, conv5_b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
    conv5 = tf.nn.relu(conv5_in + conv5_b)


    #maxpool5
    #max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
    k_h = 3; k_w = 3; s_h = 2; s_w = 2; padding = 'VALID'
    maxpool5 = max_pool(conv5, k_h, k_w, s_h, s_w, padding)

    print 'Max pool 5',maxpool5.shape
    maxpool5_flat = tf.reshape(maxpool5, [-1, 9216])

    T = tf.concat([maxpool5_flat, label],1)
    print T.shape

    G_h1 = tf.nn.relu(tf.matmul(T, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)
    print G_prob.shape

    return G_prob

