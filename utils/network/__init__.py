import numpy as np
import tensorflow as tf
from tools.ops import linear
from tools.params import param
from utils.network.ops import *


def encoder(inputs, batch_size, model_weights, lamb=1, output_dim=64, branch='g', stage='train'):
    net_data = dict(np.load(model_weights + 'reference_pretrain.npy', encoding='bytes', allow_pickle=True).item())

    scope = 'encoder.conv1.'
    kernel = param(scope + 'weights', net_data['conv1'][0])
    biases = param(scope + 'biases', net_data['conv1'][1])
    conv = tf.nn.conv2d(inputs, kernel, [1, 4, 4, 1], padding='VALID')
    out = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(out, name=scope)

    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
    radius = 2
    alpha = 2e-05
    beta = 0.75
    bias = 1.0
    lrn1 = tf.nn.local_response_normalization(pool1, depth_radius=radius, alpha=alpha, beta=beta, bias=bias,
                                              name='lrn1')

    scope = 'encoder.conv2.'
    kernel = param(scope + 'weights', net_data['conv2'][0])
    biases = param(scope + 'biases', net_data['conv2'][1])
    group = 2

    def convolve(i, k):
        return tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')

    input_groups = tf.split(lrn1, group, 3)
    kernel_groups = tf.split(kernel, group, 3)
    output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
    conv = tf.concat(output_groups, 3)
    out = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(out, name=scope)
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
    radius = 2
    alpha = 2e-05
    beta = 0.75
    bias = 1.0
    lrn2 = tf.nn.local_response_normalization(pool2, depth_radius=radius, alpha=alpha, beta=beta, bias=bias,
                                              name='lrn2')

    scope = 'encoder.conv3.'
    kernel = param(scope + 'weights', net_data['conv3'][0])
    biases = param(scope + 'biases', net_data['conv3'][1])
    conv = tf.nn.conv2d(lrn2, kernel, [1, 1, 1, 1], padding='SAME')
    out = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(out, name=scope)

    scope = 'encoder.conv4.'
    kernel = param(scope + 'weights', net_data['conv4'][0])
    biases = param(scope + 'biases', net_data['conv4'][1])
    group = 2

    def convolve(i, k):
        return tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')

    input_groups = tf.split(conv3, group, 3)
    kernel_groups = tf.split(kernel, group, 3)
    output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
    conv = tf.concat(output_groups, 3)
    out = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(out, name=scope)

    scope = 'encoder.conv5.'
    kernel = param(scope + 'weights', net_data['conv5'][0])
    biases = param(scope + 'biases', net_data['conv5'][1])
    group = 2

    def convolve(i, k):
        return tf.nn.conv2d(i, k, [1, 1, 1, 1], padding='SAME')

    input_groups = tf.split(conv4, group, 3)
    kernel_groups = tf.split(kernel, group, 3)
    output_groups = [convolve(i, k)
                     for i, k in zip(input_groups, kernel_groups)]
    conv = tf.concat(output_groups, 3)
    out = tf.nn.bias_add(conv, biases)
    conv5 = tf.nn.relu(out, name=scope)
    pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')

    scope = 'encoder.fc6.'
    shape = int(np.prod(pool5.get_shape()[1:]))
    fc6w = param(scope + 'weights', net_data['fc6'][0])
    fc6b = param(scope + 'biases', net_data['fc6'][1])
    pool5_flat = tf.reshape(pool5, [-1, shape])
    fc6l = tf.nn.bias_add(tf.matmul(pool5_flat, fc6w), fc6b)
    if stage == 'train':
        fc6 = tf.nn.dropout(tf.nn.relu(fc6l), 0.5)
    else:
        fc6 = tf.nn.relu(fc6l)

    scope = 'encoder.fc7.'
    fc7w = param(scope + 'weights', net_data['fc7'][0])
    fc7b = param(scope + 'biases', net_data['fc7'][1])
    fc7l = tf.nn.bias_add(tf.matmul(fc6, fc7w), fc7b)
    if stage == 'train':
        fc7 = tf.nn.dropout(tf.nn.relu(fc7l), 0.5)
    else:
        fc7 = tf.nn.relu(fc7l)

    fc8 = linear('encoder.d', 4096, output_dim, fc7)
    lambb = tf.add(tf.multiply(tf.cast(lamb, tf.float32), tf.constant(0.1)), tf.constant(5.0))

    if stage == "train":
        # output= tf.nn.tanh(fc8)
        # output = tan_beta(fc8, [-1.0])
        output = (2. * tf.atan(lambb * fc8)) / np.pi
        logic = linear('encoder.d1', output_dim, 10, output)
        if branch == 'g':
            f_all = output
            l_all = logic
        elif branch == 'h':
            f_p1, f_p2, l_p1, l_p2 = bran_h(pool5, batch_size, output_dim, stage='train')
            f_all = [output, f_p1, f_p2]
            l_all = [logic, l_p1, l_p2]
        else:
            f_p1, f_p2, l_p1, l_p2 = bran_v(pool5, batch_size, output_dim, stage='train')
            f_all = [output, f_p1, f_p2]
            l_all = [logic, l_p1, l_p2]
    else:
        # fc6_t = tf.nn.tanh(fc8)
        # fc6_t = tan_beta(fc8, [-1.0])
        fc6_t = (2. * tf.atan(lambb * fc8)) / np.pi
        fc6_t = tf.concat([tf.expand_dims(i, 0)
                           for i in tf.split(fc6_t, 10, 0)], 0)
        output = tf.reduce_mean(fc6_t, 0)
        logic = linear('encoder.d1', output_dim, 10, output)
        if branch == 'g':
            f_all = output
            l_all = logic
        elif branch == 'h':
            f_p1, f_p2, l_p1, l_p2 = bran_h(pool5, batch_size, output_dim, stage='val')
            f_all = [output, f_p1, f_p2]
            l_all = [logic, l_p1, l_p2]
        else:
            f_p1, f_p2, l_p1, l_p2 = bran_v(pool5, batch_size, output_dim, stage='val')
            f_all = [output, f_p1, f_p2]
            l_all = [logic, l_p1, l_p2]

    return f_all, l_all


def tan_beta(x, beta_):
    a_ = tf.multiply(beta_, x)
    one = tf.constant(1.0, dtype=tf.float32)
    ex = tf.exp(a_)
    b = tf.subtract(one, ex)
    c = tf.add(one, ex)
    ans = tf.div(b, c)
    return ans


def bran_h(pool, bs, dim, stage='train'):
    pool5_1 = pool[:, :, 0:3, :]
    pool5_2 = pool[:, :, 3:6, :]
    pool5_1 = tf.reshape(global_avg_pooling(pool5_1), (3*bs, -1))
    pool5_2 = tf.reshape(global_avg_pooling(pool5_2), (3*bs, -1))
    f_r1 = linear('encoder.r1', 256, dim, pool5_1)
    f_r2 = linear('encoder.r2', 256, dim, pool5_2)

    if stage == 'train':
        f_r1 = tf.nn.tanh(f_r1)
        f_r2 = tf.nn.tanh(f_r2)
        l_r1 = linear('encoder.r11', dim, 10, f_r1)
        l_r2 = linear('encoder.r12', dim, 10, f_r2)
    else:
        f_r1 = expend(f_r1)
        f_r2 = expend(f_r2)
        l_r1 = linear('encoder.r11', dim, 10, f_r1)
        l_r2 = linear('encoder.r12', dim, 10, f_r2)
    return f_r1, f_r2, l_r1, l_r2


def bran_v(pool, bs, dim, stage='train'):
    pool5_1 = pool[:, 0:3, :, :]
    pool5_2 = pool[:, 3:6, :, :]
    pool5_1 = tf.reshape(global_avg_pooling(pool5_1), (3*bs, -1))
    pool5_2 = tf.reshape(global_avg_pooling(pool5_2), (3*bs, -1))
    f_r01 = linear('encoder.r01', 256, dim, pool5_1)
    f_r02 = linear('encoder.r02', 256, dim, pool5_2)

    if stage == 'train':
        f_r01 = tf.nn.tanh(f_r01)
        f_r02 = tf.nn.tanh(f_r02)
        l_r01 = linear('encoder.r011', dim, 10, f_r01)
        l_r02 = linear('encoder.r012', dim, 10, f_r02)
    else:
        f_r01 = expend(f_r01)
        f_r02 = expend(f_r02)
        l_r01 = linear('encoder.r011', dim, 10, f_r01)
        l_r02 = linear('encoder.r012', dim, 10, f_r02)
    return f_r01, f_r02, l_r01, l_r02


def expend(x):
    x1 = tf.nn.tanh(x)
    x1 = tf.concat([tf.expand_dims(i, 0)
                    for i in tf.split(x1, 10, 0)], 0)
    y = tf.reduce_mean(x1, 0)
    return y