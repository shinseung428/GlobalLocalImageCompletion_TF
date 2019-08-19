import tensorflow as tf


def batch_norm(input, name="batch_norm"):
    with tf.variable_scope(name) as scope:
        input = tf.identity(input)
        channels = input.get_shape()[3]

        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))

        mean, variance = tf.nn.moments(input, axes=[0,1,2], keep_dims=False)

        normalized_batch = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=1e-5)

        return normalized_batch


def linear(input, output_size, name="linear"):
    shape = input.get_shape().as_list()

    with tf.variable_scope(name) as scope:
        matrix = tf.get_variable("W", [shape[1], output_size], tf.float32, tf.random_normal_initializer(stddev=0.02))
        bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(0.0))

        return tf.matmul(input, matrix) + bias


def conv2d(input, out_filter, padding, kernel=5, stride=2, name="conv2d"):
    input_shape = input.get_shape().as_list()
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", [kernel, kernel, input_shape[-1], out_filter], initializer=tf.random_normal_initializer(stddev=0.02))
        b = tf.get_variable("b", [out_filter], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input, w,
                            strides=[1, stride, stride, 1],
                            padding=padding
                            )

        conv = tf.reshape(tf.nn.bias_add(conv, b), conv.get_shape())

        return conv

def deconv2d(input, out_shape, name="deconv2d"):
    input_shape = input.get_shape().as_list()
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", [4, 4, out_shape[-1], input_shape[-1]], initializer=tf.random_normal_initializer(stddev=0.02))
        b = tf.get_variable("b", [out_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.nn.conv2d_transpose(input, w,
                                        output_shape=out_shape,
                                        strides=[1, 2, 2, 1],
                                        padding="SAME")
        deconv = tf.reshape(tf.nn.bias_add(deconv, b), deconv.get_shape())

        return deconv

def dilate_conv2d(input, out_shape, rate, name="dilate_conv2d"):
    input_shape = input.get_shape().as_list()
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", [3, 3, input_shape[-1], out_shape[-1]])
        b = tf.get_variable("b", [out_shape[-1]], initializer=tf.constant_initializer(0.0))
    dilate_conv = tf.nn.atrous_conv2d(input, w,
                                      rate=rate,
                                      padding="SAME"
                                      )
    dilate_conv = tf.reshape(tf.nn.bias_add(dilate_conv, b), dilate_conv.get_shape())

    return dilate_conv
