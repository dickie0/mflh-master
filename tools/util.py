import tensorflow as tf


def resize_img(inputs, width_height):
    img = (inputs + 1.) * 255.99 / 2
    reshaped_image = tf.cast(img, tf.float32)
    reshaped_image = tf.reshape(
        reshaped_image, [-1, 3, width_height, width_height])
    transpose_image = tf.transpose(reshaped_image, perm=[0, 2, 3, 1])
    resized_image = tf.image.resize_bilinear(transpose_image, [64, 64])
    return resized_image


def scalar_summary(tag, value):
    return tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
