""" Necessary functions to build the model"""
import tensorflow as tf
import tensorflow.keras as keras
import os
from model.utils import *

tf.random.set_seed(1231)

weight_init = tf.initializers.TruncatedNormal(mean=0.0, stddev=0.02)
weight_regularizer = orthogonal_regularizer(0.0001)
weight_regularizer_fully = orthogonal_regularizer_fully(0.0001)


def Resize2DBilinear(size):
    return keras.Lambda(lambda x: tf.image.resize(x, size, method=tf.image.ResizeMethod.BILINEAR, align_corners=True))


def Resize2DNearest(size):
    return keras.Lambda(lambda x: tf.image.resize(x, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, align_corners=True))

# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# Loss
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+


# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# Sampling
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2])

    return gap


def global_sum_pooling(x):
    gsp = tf.reduce_sum(x, axis=[1, 2])

    return gsp


def max_pooling():
    return keras.layers.MaxPooling2D(pool_size=2, strides=2, padding='SAME')


def up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize(x, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, size=new_size)


# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# Layers
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
class Conv(keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel=3,
                 stride=1,
                 pad=0,
                 pad_type='zero',
                 sn=False,
                 use_bias=True,
                 **kwargs):
        super(Conv, self).__init__(**kwargs)
        self.filters = filters
        self.kernel = kernel
        self.stride = stride
        self.pad = pad
        self.pad_type = pad_type
        self.sn = sn
        if self.sn:
            self.spectral_norm = SpectralNorm()
        self.use_bias = use_bias
        self.conv2d = None

    def build(self, input_shape):
        if self.sn:
            self.w = self.add_variable(
                shape=[self.kernel, self.kernel, input_shape[-1], self.filters],
                initializer=weight_init,
                regularizer=weight_regularizer,
                name='kernel',
            )

            if self.use_bias:
                self.bias = self.add_variable(name='bias',
                                              shape=[self.filters],
                                              initializer=keras.initializers.get('zeros'))

        else:
            self.conv2d = keras.layers.Conv2D(
                filters=self.filters,
                kernel_size=self.kernel,
                kernel_initializer=weight_init,
                kernel_regularizer=weight_regularizer,
                strides=self.stride,
                use_bias=self.use_bias)

        super(Conv, self).build(input_shape)

    def call(self, inputs):
        if self.pad > 0:
            h = inputs.get_shape()[1]
            if h % self.stride == 0:
                pad = self.pad * 2
            else:
                pad = max(self.kernel - (h % self.stride), 0)

            pad_top = pad // 2
            pad_bottom = pad - pad_top
            pad_left = pad // 2
            pad_right = pad - pad_left

            if self.pad_type == 'zero':
                inputs = tf.pad(inputs, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]])
            elif self.pad_type == 'reflect':
                inputs = tf.pad(inputs, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], mode='REFLECT')
            else:
                raise Exception("Not a valid type")

        if self.sn:
            inputs = tf.nn.conv2d(inputs,
                                  filters=self.spectral_norm(self.w),
                                  strides=[1, self.stride, self.stride, 1],
                                  padding='VALID')
            if self.use_bias:
                inputs = tf.nn.bias_add(inputs, self.bias)

        else:
            inputs = self.conv2d(inputs)

        return inputs


class DeConv(keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel=3,
                 stride=1,
                 padding='SAME',
                 sn=False,
                 use_bias=True,
                 **kwargs):
        super(DeConv, self).__init__(**kwargs)
        self.filters = filters
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.sn = sn
        if self.sn:
            self.spectral_norm = SpectralNorm()
        self.use_bias = use_bias
        self.w = None
        self.bias = None
        self.deconv = None

    def build(self, input_shape):
        if self.sn:
            self.w = self.add_variable(
                shape=[self.kernel, self.kernel, input_shape[-1], self.filters],
                initializer=weight_init,
                regularizer=weight_regularizer,
                name='kernel'
            )

            if self.use_bias:
                self.bias = self.add_variable(name='bias',
                                              shape=[self.filters],
                                              initializer=keras.initializers.get('zeros'))

        else:
            self.deconv = keras.layers.Conv2DTranspose(
                filters=self.filters,
                kernel_size=self.kernel,
                kernel_initializer=weight_init,
                kernel_regularizer=weight_regularizer,
                strides=self.stride,
                padding=self.padding,
                use_bias=self.use_bias)

        super(DeConv, self).build(input_shape)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        if self.padding == 'SAME':
            output_shape = [input_shape[0],
                            input_shape[1] * self.stride,
                            input_shape[2] * self.stride,
                            self.filters]
        else:
            output_shape = [input_shape[0],
                            input_shape[1] * self.stride + max(self.kernel - self.stride, 0),
                            input_shape[2] * self.stride + max(self.kernel - self.stride, 0),
                            self.filters]
        if self.sn:
            inputs = tf.nn.conv2d_transpose(inputs,
                                            filters=self.spectral_norm(self.w),
                                            output_shape=output_shape,
                                            strides=[1, self.stride, self.stride, 1],
                                            padding=self.padding)
            if self.use_bias:
                inputs = tf.nn.bias_add(inputs, self.bias)

        else:
            inputs = self.deconv(inputs)

        return inputs


class Linear(keras.layers.Layer):

    def __init__(self, units, sn=False, use_bias=True, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.units = units
        self.sn = sn
        if self.sn:
            self.sn = SpectralNorm()
        self.use_bias = use_bias
        self.dense = None

    def build(self, input_shape):
        filters = input_shape[-1]
        if self.sn:
            self.w = self.add_variable(name="kernel",
                                       shape=[filters, self.units],
                                       dtype='tf.float32',
                                       initializer=weight_init,
                                       regularizer=weight_regularizer)
            if self.use_bias:
                self.bias = self.add_variable(name="bias",
                                         shape=[self.units],
                                         initializer=keras.initializers.get('zeros'))
        else:
            self.dense = keras.layers.Dense(units=self.units,
                                          kernel_initializer=weight_init,
                                          kernel_regularizer=weight_regularizer_fully,
                                          use_bias=self.use_bias)

        super(Linear, self).build(input_shape)

    def call(self, inputs):
        if self.sn:
            inputs = tf.matmul(inputs, self.spectral_norm(self.w))
            if self.use_bias:
                inputs = inputs + self.bias
        else:
            inputs = self.dense(inputs)

        return inputs


def hw_flatten(x):
    return tf.reshape(x, shape=tf.convert_to_tensor([x.get_shape()[0],
                                x.get_shape()[1] * x.get_shape()[2],
                                x.get_shape()[3]]))


# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# Conditional Batch Normalization
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
class ClassConditionalBatchNorm(tf.keras.layers.Layer):

    def __init__(self, z, **kwargs):
        super(ClassConditionalBatchNorm, self).__init__(**kwargs, dynamic=True)
        self.test_mean = None
        self.test_var = None
        self.ema_mean = None
        self.ema_var = None
        self.z = z

    def build(self, input_shape):
        c = input_shape[-1]
        zeros = tf.keras.initializers.Constant(0.0)
        self.test_mean = self.add_variable(
            initializer=zeros,
            shape=[c],
            name='allocated_mean',
            dtype=tf.float32,
            trainable=False)

        one = tf.keras.initializers.Constant(1.0)
        self.test_var = self.add_variable(
            initializer=one,
            shape=[c],
            name='allocated_var',
            dtype=tf.float32,
            trainable=False)

        super(ClassConditionalBatchNorm, self).build(input_shape)

    def call(self, x, is_training=True):
        _, _, _, c = x.get_shape().as_list()
        decay = 0.9
        eps = 1e-05

        split = tf.keras.layers.Flatten()(self.z)
        beta = Linear(units=c)(split)
        gamma = Linear(units=c)(split)

        beta = tf.reshape(beta, shape=[-1, 1, 1, c])
        gamma = tf.reshape(gamma, shape=[-1, 1, 1, c])

        if is_training:
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
            self.ema_mean = self.test_mean * decay + batch_mean * (1 - decay)
            self.ema_var = self.test_var * decay + batch_mean * (1 - decay)

            with tf.control_dependencies([self.ema_mean, self.ema_var]):
                return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, eps)
        else:
            return tf.nn.batch_normalization(x, self.test_mean, self.test_var, beta, gamma, eps)

    def compute_output_shape(self, input_shape):
        return input_shape


# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# Model blocks
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
class ResBlock(keras.Model):
    """
    Create a ResNet block for up sampling
    """
    def __init__(self, z, channel, use_bias=True, sn=True):
        super(ResBlock, self).__init__(name='')
        self.channel = channel
        self.use_bias = use_bias
        self.sn = sn
        self.z = z
        self.conv1 = Conv(filters=self.channel, kernel=3, stride=1, pad=1, sn=self.sn)
        self.conv2 = Conv(filters=self.channel, kernel=3, stride=1, pad=1, sn=self.sn)
        # self.conv3 = Conv(filters=self.channel, kernel=1, stride=2, pad=1, sn=self.sn)
        self.ccbn = ClassConditionalBatchNorm(self.z)

    def call(self, inputs, is_training=True):
        x = self.conv1(inputs)
        x = self.ccbn(x)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.ccbn(x)

        return x + inputs


class ResBlockUp(keras.Model):
    """
    Create a ResNet block for up sampling
    """
    def __init__(self, z, channel, use_bias=True, sn=True):
        super(ResBlockUp, self).__init__(name='')
        self.channel = channel
        self.use_bias = use_bias
        self.sn = sn
        self.z = z
        self.deconv1 = DeConv(filters=self.channel, kernel=3, stride=2, sn=self.sn, use_bias=self.use_bias)
        self.deconv2 = DeConv(filters=self.channel, kernel=3, stride=1, sn=self.sn, use_bias=self.use_bias)
        self.deconv3 = DeConv(filters=self.channel, kernel=1, stride=2, sn=self.sn, use_bias=self.use_bias, padding='VALID')
        self.ccbn = ClassConditionalBatchNorm(z=self.z)

    def call(self, inputs, is_training=True):
        x = self.ccbn(inputs)
        x = tf.nn.relu(x)
        x = self.deconv1(x)

        x = self.ccbn(x)
        x = tf.nn.relu(x)
        x = self.deconv2(x)

        x_init = self.deconv3(inputs)

        return x + x_init


class ResBlockDown(keras.Model):
    """
    Create a ResNet block for down sampling
    """
    def __init__(self, z, channel, use_bias=True, sn=True):
        super(ResBlockDown, self).__init__(name='')
        self.channel = channel
        self.use_bias = use_bias
        self.sn = sn
        self.z = z
        self.conv1 = Conv(filters=self.channel, kernel=3, stride=2, sn=self.sn, use_bias=self.use_bias, pad=1)
        self.conv2 = Conv(filters=self.channel, kernel=3, stride=1, sn=self.sn, use_bias=self.use_bias, pad=1)
        self.conv3 = Conv(filters=self.channel, kernel=1, stride=2, sn=self.sn, use_bias=self.use_bias, pad=0)
        self.bn = batch_norm()

    def call(self, inputs, is_training=True):
        x = self.bn(inputs)
        x = tf.nn.relu(x)
        x = self.conv1(x)
        print(x.get_shape())

        x = self.bn(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        print(x.get_shape())

        x_init = self.conv3(inputs)
        print(x.get_shape())

        return x + x_init


class SelfAttention(keras.Model):
    def __init__(self, channels, sn=True):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.sn = sn
        self.f = Conv(filters=channels//8, kernel=1, stride=1, sn=sn)
        self.g = Conv(filters=channels//8, kernel=1, stride=1, sn=sn)
        self.h = Conv(filters=channels//2, kernel=1, stride=1, sn=sn)
        self.o = Conv(filters=channels, kernel=1, stride=1, sn=sn)
        self.mp = max_pooling()

    def call(self, inputs):
        f = self.f(inputs)
        f = self.mp(f)

        g = self.g(inputs)

        h = self.h(inputs)
        g = self.mp(h)

        print(g.get_shape())
        g_ = hw_flatten(g)
        print(g_.get_shape())
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)

        beta = tf.nn.softmax(s)
        shape = tf.shape(inputs)

        o = tf.matmul(beta, hw_flatten(h))
        o = tf.reshape(o, shape=[shape[0], shape[1], shape[2], self.channels // 2])
        o = self.o(o)
        zeros = keras.initializers.Constant(0.0)
        gamma = tf.Variable(lambda: zeros(shape=[1]))
        x = gamma * o + inputs
        # inputs = o + inputs

        return inputs


if __name__ == '__main__':
    new_weights = tf.ones((3, 256, 256, 3), dtype=tf.float32)
    dummy_out = tf.random.normal((3, 1))

    z = tf.split(new_weights, num_or_size_splits=[256//8] * 8, axis=1)
    x_init = tf.keras.layers.Flatten()(z[0])
    x = Linear(units=16*16*16*1)(x_init)
    x = tf.reshape(x, shape=[-1, 16, 16, 16])
    model = SelfAttention(channels=16, sn=False)
    # out = model(x)
    # print(out)
    # print(model.summary())
    logdir = "../logs/graph/"
    tf.summary.trace_on(graph=True, profiler=True)
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, write_graph=False)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    print("the input before fit: ", x.get_shape())
    model.fit(
        x,
        dummy_out,
        batch_size=1,
        epochs=1,
        callbacks=[tensorboard_callback]
    )

