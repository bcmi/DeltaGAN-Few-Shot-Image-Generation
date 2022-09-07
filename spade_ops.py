import tensorflow as tf
import tensorflow.contrib as tf_contrib
from utils import pytorch_xavier_weight_factor, pytorch_kaiming_weight_factor

##################################################################################
# Initialization
##################################################################################

"""
pytorch xavier (gain)
https://pytorch.org/docs/stable/_modules/torch/nn/init.html
if uniform :
    factor = gain * gain
    mode = 'FAN_AVG'
else :
    SPADE use, gain=0.02
    factor = (gain * gain) / 1.3
    mode = 'FAN_AVG'
    
pytorch : trunc_stddev = gain * sqrt(2 / (fan_in + fan_out))
tensorflow  : trunc_stddev = sqrt(1.3 * factor * 2 / (fan_in + fan_out))
"""

"""
pytorch kaiming (a=0)
https://pytorch.org/docs/stable/_modules/torch/nn/init.html
if uniform :
    a = 0 -> gain = sqrt(2)
    factor = gain * gain
    mode='FAN_IN'
else :
    a = 0 -> gain = sqrt(2)
    factor = (gain * gain) / 1.3
    mode = 'FAN_OUT', but SPADE use 'FAN_IN'
    
pytorch : trunc_stddev = gain * sqrt(2 / (fan_in + fan_out))
tensorflow  : trunc_stddev = sqrt(1.3 * factor * 2 / (fan_in + fan_out))
"""

factor, mode, uniform = pytorch_xavier_weight_factor(gain=0.02, uniform=False)
weight_init = tf_contrib.layers.variance_scaling_initializer(factor=factor, mode=mode, uniform=uniform)
# tf.truncated_normal_initializer(mean=0.0, stddev=0.02)

weight_regularizer = None
weight_regularizer_fully = None

# Normal : tf.random_normal_initializer(mean=0.0, stddev=0.02)
# Truncated_normal : tf.truncated_normal_initializer(mean=0.0, stddev=0.02)
# Orthogonal : tf.orthogonal_initializer(0.02)

##################################################################################
# Regularization
##################################################################################

# l2_decay : tf.contrib.layers.l2_regularizer(0.0001)
# orthogonal_regularizer : orthogonal_regularizer(0.0001) # orthogonal_regularizer_fully(0.0001)


##################################################################################
# Layer
##################################################################################

def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv_0'):
    with tf.variable_scope(scope):
        if pad > 0:
            h = x.get_shape().as_list()[1]
            if h % stride == 0:
                pad = pad * 2
            else:
                pad = max(kernel - (h % stride), 0)

            pad_top = pad // 2
            pad_bottom = pad - pad_top
            pad_left = pad // 2
            pad_right = pad - pad_left

            if pad_type == 'zero':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
            if pad_type == 'reflect':
                x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')

        if sn:
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='VALID')
            if use_bias:
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else:
            x = tf.layers.conv2d(inputs=x, filters=channels,
                                 kernel_size=kernel, kernel_initializer=weight_init,
                                 kernel_regularizer=weight_regularizer,
                                 strides=stride, use_bias=use_bias)

        return x


def partial_conv(x, channels, kernel=3, stride=2, use_bias=True, padding='SAME', sn=False, scope='conv_0'):
    with tf.variable_scope(scope):
        if padding.lower() == 'SAME'.lower():
            with tf.variable_scope('mask'):
                _, h, w, _ = x.get_shape().as_list()

                slide_window = kernel * kernel
                mask = tf.ones(shape=[1, h, w, 1])

                update_mask = tf.layers.conv2d(mask, filters=1,
                                               kernel_size=kernel, kernel_initializer=tf.constant_initializer(1.0),
                                               strides=stride, padding=padding, use_bias=False, trainable=False)

                mask_ratio = slide_window / (update_mask + 1e-8)
                update_mask = tf.clip_by_value(update_mask, 0.0, 1.0)
                mask_ratio = mask_ratio * update_mask

            with tf.variable_scope('x'):
                if sn:
                    w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels],
                                        initializer=weight_init, regularizer=weight_regularizer)
                    x = tf.nn.conv2d(input=x, filter=spectral_norm(w), strides=[1, stride, stride, 1], padding=padding)
                else:
                    x = tf.layers.conv2d(x, filters=channels,
                                         kernel_size=kernel, kernel_initializer=weight_init,
                                         kernel_regularizer=weight_regularizer,
                                         strides=stride, padding=padding, use_bias=False)
                x = x * mask_ratio

                if use_bias:
                    bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))

                    x = tf.nn.bias_add(x, bias)
                    x = x * update_mask
        else:
            if sn:
                w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels],
                                    initializer=weight_init, regularizer=weight_regularizer)
                x = tf.nn.conv2d(input=x, filter=spectral_norm(w), strides=[1, stride, stride, 1], padding=padding)
                if use_bias:
                    bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))

                    x = tf.nn.bias_add(x, bias)
            else:
                x = tf.layers.conv2d(x, filters=channels,
                                     kernel_size=kernel, kernel_initializer=weight_init,
                                     kernel_regularizer=weight_regularizer,
                                     strides=stride, padding=padding, use_bias=use_bias)

        return x

def fully_connected(x, units, use_bias=True, sn=False, scope='linear'):
    with tf.variable_scope(scope):
        x = flatten(x)
        shape = x.get_shape().as_list()
        channels = shape[-1]

        if sn:
            w = tf.get_variable("kernel", [channels, units], tf.float32,
                                initializer=weight_init, regularizer=weight_regularizer_fully)
            if use_bias:
                bias = tf.get_variable("bias", [units],
                                       initializer=tf.constant_initializer(0.0))

                x = tf.matmul(x, spectral_norm(w)) + bias
            else:
                x = tf.matmul(x, spectral_norm(w))

        else:
            x = tf.layers.dense(x, units=units, kernel_initializer=weight_init,
                                kernel_regularizer=weight_regularizer_fully,
                                use_bias=use_bias)

        return x

def flatten(x):
    return tf.layers.flatten(x)


##################################################################################
# Residual-block
##################################################################################

def spade_resblock(segmap, x_init, channels, use_bias=True, sn=False, scope='spade_resblock'):
    channel_in = x_init.get_shape().as_list()[-1]
    channel_middle = min(channel_in, channels)

    with tf.variable_scope(scope) :
        x = spade(segmap, x_init, channel_in, use_bias=use_bias, sn=False, scope='spade_1')
        x = lrelu(x, 0.2)
        x = conv(x, channels=channel_middle, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn, scope='conv_1')

        x = spade(segmap, x, channels=channel_middle, use_bias=use_bias, sn=False, scope='spade_2')
        x = lrelu(x, 0.2)
        x = conv(x, channels=channels, kernel=3, stride=1, pad=1, use_bias=use_bias, sn=sn, scope='conv_2')

        if channel_in != channels :
            x_init = spade(segmap, x_init, channels=channel_in, use_bias=use_bias, sn=False, scope='spade_shortcut')
            x_init = conv(x_init, channels=channels, kernel=1, stride=1, use_bias=False, sn=sn, scope='conv_shortcut')

        return x + x_init


def spade(segmap, x_init, channels, use_bias=True, sn=False, scope='spade') :
    with tf.variable_scope(scope) :
        x = param_free_norm(x_init)

        _, x_h, x_w, _ = x_init.get_shape().as_list()
        _, segmap_h, segmap_w, _ = segmap.get_shape().as_list()

        factor_h = segmap_h // x_h  # 256 // 4 = 64
        factor_w = segmap_w // x_w

        segmap_down = down_sample(segmap, factor_h, factor_w)

        segmap_down = conv(segmap_down, channels=128, kernel=5, stride=1, pad=2, use_bias=use_bias, sn=sn, scope='conv_128')
        segmap_down = relu(segmap_down)

        segmap_gamma = conv(segmap_down, channels=channels, kernel=5, stride=1, pad=2, use_bias=use_bias, sn=sn, scope='conv_gamma')
        segmap_beta = conv(segmap_down, channels=channels, kernel=5, stride=1, pad=2, use_bias=use_bias, sn=sn, scope='conv_beta')

        x = x * (1 + segmap_gamma) + segmap_beta

        return x

def param_free_norm(x, epsilon=1e-5) :
    x_mean, x_var = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
    x_std = tf.sqrt(x_var + epsilon)

    return (x - x_mean) / x_std

##################################################################################
# Sampling
##################################################################################

def resize_256(x) :
    return tf.image.resize_bilinear(x, size=[256, 256])

def up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize_bilinear(x, size=new_size)

def down_sample(x, scale_factor_h, scale_factor_w) :
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h // scale_factor_h, w // scale_factor_w]

    return tf.image.resize_nearest_neighbor(x, size=new_size)

def down_sample_avg(x, scale_factor=2) :
    return tf.layers.average_pooling2d(x, pool_size=3, strides=scale_factor, padding='SAME')


##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.01):
    # pytorch alpha is 0.01
    return tf.nn.leaky_relu(x, alpha)

def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.tanh(x)


##################################################################################
# Normalization function
##################################################################################

def instance_norm(x, scope='instance_norm'):
    return tf_contrib.layers.instance_norm(x,
                                           epsilon=1e-05,
                                           center=True, scale=True,
                                           scope=scope)

def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm