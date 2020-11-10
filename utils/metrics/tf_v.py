import tensorflow as tf
import sys 
import numpy as np


def norm(x, keepdims=False):
    return tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1, keepdims=keepdims))


def normed(x):
    return x / norm(x, keepdims=True)


def euclidean2(x1, x2):
    return tf.reduce_sum(tf.square(x1 - x2), axis=-1)


def euclidean(x1, x2):
    return tf.sqrt(euclidean2(x1, x2))


def averaged_euclidean2(x1, x2):
    return tf.reduce_mean(tf.square(x1 - x2), axis=-1)


def averaged_euclidean(x1, x2):
    return tf.sqrt(averaged_euclidean2(x1, x2)) 


def normed_euclidean2(x1, x2):
    return euclidean2(normed(x1), normed(x2))


def inner_product(x1, x2):
    return - tf.reduce_sum(x1 * x2, axis=-1) 


def cosine(x1, x2):
    return (1 + inner_product(normed(x1), normed(x2))) / 2


def metric(x1, x2=None, pair=True, dist_type="euclidean2"):
    if x2 is None:
        x2 = x1
    if pair:
        x1 = tf.expand_dims(x1, 1)
        x2 = tf.expand_dims(x2, 0)
    return getattr(sys.modules[__name__], dist_type)(x1, x2)