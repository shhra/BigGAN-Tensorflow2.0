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
                                  filters=spectral_norm(self.w),
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
        self.use_bias = use_bias
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
        input_shape = inputs.get_shape()
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
                                            filters=spectral_norm(self.w),
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
            inputs = tf.matmul(inputs, spectral_norm(self.w))
            if self.use_bias:
                inputs = inputs + self.bias
        else:
            inputs = self.dense(inputs)

        return inputs


def hw_flatten(x):
    return tf.reshape(x, shape=[x.get_shape()[0], -1, x.get_shape()[-1]])

# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# Model blocks
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+


class ResBlock(keras.Model):
    """
    Create a ResNet block for up sampling
    """
    def __init__(self, inputs, channel):
        super(ResBlock, self).__init__(name='')
        pass


class ResBlockUp(keras.Model):
    """
    Create a ResNet block for up sampling
    """
    def __init__(self, inputs, channel):
        super(ResBlockUp, self).__init__(name='')
        pass


class ResBlockDown(keras.Model):
    """
    Create a ResNet block for down sampling
    """
    def __init__(self, inputs, channels, kernel):
        super(ResBlockDown, self).__init__(name='')
        pass


class DummyModel(keras.Model):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.l1 = Conv(filters=16, kernel=3, stride=1, pad=0, sn=False)
        self.l2 = DeConv(filters=16, kernel=3, stride=1, padding='VALID', sn=False)
        self.flat = keras.layers.Flatten()
        self.final = Linear(1)

    def call(self, x):
        x = self.l1(x)
        x = tf.nn.relu(x)
        x = self.l2(x)
        x = tf.nn.relu(x)
        x = self.flat(x)
        x = tf.nn.relu(x)
        x = self.final(x)

        return x


if __name__ == '__main__':
    new_weights = tf.ones((2000, 16, 16, 64))
    dummy_out = tf.ones((2000, 1))

    input_data = keras.Input(shape=(3, 3, 1))
    print("The shape of input is: {}".format(input_data.get_shape()))

    model = DummyModel()
    logdir = "../logs/graph/"
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    model.build(input_shape=new_weights.get_shape())
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.fit(
        new_weights,
        dummy_out,
        batch_size=1000,
        epochs=1,
        callbacks=[tensorboard_callback]
    )
    model.summary()


