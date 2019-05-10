""" Necessary functions to build the model"""
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras.utils import conv_utils
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
def discriminator_loss(loss_func, real, fake):
    real_loss = 0
    fake_loss = 0

    if loss_func.__contains__('wgan'):
        real_loss = -tf.reduce_mean(real)
        fake_loss = tf.reduce_mean(fake)

    if loss_func == 'lsgan':
        real_loss = tf.reduce_mean(tf.math.squared_difference(real, 1.0))
        fake_loss = tf.reduce_mean(tf.square(fake))

    if loss_func == 'gan' or loss_func == 'dragan':
        real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(real), logits=real))
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(fake), logits=fake))

    if loss_func == 'hinge':
        real_loss = tf.reduce_mean(tf.nn.relu(1.0 - real))
        fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake))

    loss = real_loss + fake_loss

    return loss


def generator_loss(loss_func, fake):
    fake_loss = 0

    if loss_func.__contains__('wgan'):
        fake_loss = -tf.reduce_mean(fake)

    if loss_func == 'lsgan':
        fake_loss = tf.reduce_mean(tf.math.squared_difference(fake, 1.0))

    if loss_func == 'gan' or loss_func == 'dragan':
        fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(fake), logits=fake))

    if loss_func == 'hinge':
        fake_loss = -tf.reduce_mean(fake)

    loss = fake_loss

    return loss


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


def avg_pooling():
    return keras.layers.AveragePooling2D(pool_size=2, strides=2, padding='SAME')


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
                 is_training=True,
                 **kwargs):
        super(Conv, self).__init__(**kwargs, dynamic=True)
        self.filters = filters
        self.kernel = kernel
        self.stride = stride
        self.pad = pad
        if pad == 0:
            self.padding = 'valid'
        else:
            self.padding = 'same'
        self.pad_type = pad_type
        self.sn = sn
        if self.sn:
            self.spectral_norm = SpectralNorm()
        self.use_bias = use_bias
        self.conv2d = None
        self.is_training = is_training
        # self.w = None
        # self.bias = None

    def build(self, input_shape):
        if self.sn:
            self.w = self.add_variable(
                shape=[self.kernel, self.kernel, input_shape[-1], self.filters],
                initializer=weight_init,
                regularizer=weight_regularizer,
                name='kernel',
                trainable=self.is_training
            )

            if self.use_bias:
                self.bias = self.add_variable(name='bias',
                                              shape=[self.filters],
                                              initializer=keras.initializers.get('zeros'),
                                              trainable=self.is_training)

        else:
            self.conv2d = keras.layers.Conv2D(
                filters=self.filters,
                kernel_size=self.kernel,
                kernel_initializer=weight_init,
                kernel_regularizer=weight_regularizer,
                strides=self.stride,
                use_bias=self.use_bias,
                trainable=self.is_training)

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

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel,
                padding=self.padding,
                stride=self.stride)
            new_space.append(new_dim)
        return tf.TensorShape([input_shape[0]] + new_space + [self.filters])


class DeConv(keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel=3,
                 stride=1,
                 padding='SAME',
                 sn=False,
                 use_bias=True,
                 is_training=True,
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
        self.is_training = is_training

    def build(self, input_shape):
        if self.sn:
            self.w = self.add_variable(
                shape=[self.kernel, self.kernel, input_shape[-1], self.filters],
                initializer=weight_init,
                regularizer=weight_regularizer,
                name='kernel',
                trainable=self.is_training
            )

            if self.use_bias:
                self.bias = self.add_variable(name='bias',
                                              shape=[self.filters],
                                              initializer=keras.initializers.get('zeros'),
                                              trainable=self.is_training)

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
        # print("The input shape from {} is {}".format(self.name, inputs.shape))
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
        # print("The output shape from {} is {}".format(self.name, output_shape))
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

    def __init__(self, units, sn=False, use_bias=True, is_training=True, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.units = units
        self.sn = sn
        if self.sn:
            self.spectral_norm = SpectralNorm()
        self.use_bias = use_bias
        self.dense = None
        self.is_training = is_training

    def build(self, input_shape):
        filters = input_shape[-1]
        if self.sn:
            self.w = self.add_variable(name="kernel",
                                       shape=[filters, self.units],
                                       initializer=weight_init,
                                       regularizer=weight_regularizer,
                                       trainable=self.is_training)
            if self.use_bias:
                self.bias = self.add_variable(name="bias",
                                         shape=[self.units],
                                         initializer=keras.initializers.get('zeros'),
                                         trainable=self.is_training)
        else:
            self.dense = keras.layers.Dense(units=self.units,
                                          kernel_initializer=weight_init,
                                          kernel_regularizer=weight_regularizer_fully,
                                          use_bias=self.use_bias, trainable=self.is_training)

        super(Linear, self).build(input_shape)

    def call(self, inputs):
        if self.sn:
            inputs = tf.matmul(inputs, self.spectral_norm(self.w))
            if self.use_bias:
                inputs = inputs + self.bias
        else:
            inputs = self.dense(inputs)

        return inputs

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([self.units])


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
        c = tf.shape(x)[-1]
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
    def __init__(self, channels, use_bias=True, sn=True):
        super(ResBlock, self).__init__(name='')
        self.channel = channels
        self.use_bias = use_bias
        self.sn = sn
        self.conv1 = Conv(filters=self.channel, kernel=3, stride=1, pad=1, sn=self.sn, is_training=True)
        self.conv2 = Conv(filters=self.channel, kernel=3, stride=1, pad=1, sn=self.sn, is_training=True)
        self.bn = batch_norm()

    def call(self, inputs, is_training=True):
        x = self.conv1(inputs)
        x = self.bn(x)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn(x)

        return x + inputs

    def compute_output_shape(self, input_shape):
        return self.conv2.compute_output_shape(input_shape)


class ResBlockUp(keras.Model):
    """
    Create a ResNet block for up sampling
    """
    def __init__(self, z, channels, use_bias=True, sn=False, is_training=True):
        super(ResBlockUp, self).__init__(name='')
        self.channel = channels
        self.is_training=is_training
        self.use_bias = use_bias
        self.sn = sn
        self.z = z
        self.conv1 = Conv(filters=self.channel, kernel=3, stride=1, sn=self.sn, use_bias=self.use_bias, pad=1, is_training=self.is_training)
        self.conv2 = Conv(filters=self.channel, kernel=3, stride=1, sn=self.sn, use_bias=self.use_bias, pad=1, is_training=self.is_training)
        self.conv3 = Conv(filters=self.channel, kernel=1, stride=1, sn=self.sn, use_bias=self.use_bias, is_training=True)
        self.ccbn1 = ClassConditionalBatchNorm(z=self.z)
        self.ccbn2 = ClassConditionalBatchNorm(z=self.z)
        self.relu = keras.layers.ReLU()
        self.final_shape = None

    def call(self, inputs, is_training=True):
        x = self.ccbn1(inputs)
        x = self.relu(x)
        x = up_sample(x)
        x_init = up_sample(inputs)
        x = self.conv1(x)
        x = self.ccbn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x_init = self.conv3(x_init)

        return x + x_init

    def compute_output_shape(self, input_shape):
        return self.conv3.compute_output_shape(input_shape)


class ResBlockDown(keras.Model):
    """
    Create a ResNet block for down sampling
    """
    def __init__(self, channels, use_bias=True, sn=True, is_training=True):
        super(ResBlockDown, self).__init__(name='')
        self.channel = channels
        self.use_bias = use_bias
        self.sn = sn
        self.is_training = is_training
        self.conv1 = Conv(filters=self.channel, kernel=3, stride=1, sn=self.sn, use_bias=self.use_bias, pad=1, is_training=self.is_training)
        self.conv2 = Conv(filters=self.channel, kernel=3, stride=1, sn=self.sn, use_bias=self.use_bias, pad=1, is_training=self.is_training)
        self.avg1 = avg_pooling()
        self.avg2 = avg_pooling()
        self.conv3 = Conv(filters=self.channel, kernel=1, stride=1, sn=self.sn, use_bias=self.use_bias, pad=0, is_training=self.is_training)
        self.conv4 = Conv(filters=self.channel, kernel=1, stride=2, sn=self.sn, use_bias=self.use_bias, pad=0, is_training=False)
        self.bn1 = batch_norm()
        self.bn2 = batch_norm()
        self.relu = keras.layers.ReLU()

    def call(self, inputs):
        x = self.relu(inputs)
        x = self.conv1(x)

        x = self.relu(x)
        x = self.conv2(x)
        x = self.avg1(x)

        x_init = self.conv3(inputs)
        x_init = self.avg2(x_init)

        return x + x_init

    def compute_output_shape(self, input_shape):
        return self.conv4.compute_output_shape(input_shape)


class SelfAttention(keras.Model):
    def __init__(self, channels, sn=False, is_training=True):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.sn = sn
        self.is_training = is_training
        self.f = Conv(filters=channels//8, kernel=1, stride=1, pad=0, sn=sn, is_training=self.is_training)
        self.g = Conv(filters=channels//8, kernel=1, stride=1, pad=0, sn=sn, is_training=self.is_training)
        self.h = Conv(filters=channels, kernel=1, stride=1, pad=0, sn=sn, is_training=self.is_training)
        self.o = Conv(filters=channels, kernel=1, stride=1, pad=0, sn=sn, is_training=self.is_training)
        zeros = keras.initializers.Constant(0.0)
        self.gamma = tf.Variable(lambda: zeros(shape=[1]))

    def call(self, inputs):
        @tf.function
        def hw_flatten(x):
            return tf.reshape(x, shape=[tf.shape(x)[0], tf.shape(x)[1]*tf.shape(x)[2], tf.shape(x)[3]])
        shape = tf.shape(inputs)
        f = self.f(inputs)
        g = self.g(inputs)
        h = self.h(inputs)
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)
        beta = tf.nn.softmax(s)
        o = tf.matmul(beta, hw_flatten(h))
        o = tf.reshape(o, shape=shape)
        o = self.o(o)
        x = self.gamma * o + inputs

        return x

    def compute_output_shape(self, input_shape):
        return self.o.compute_output_shape(input_shape)


if __name__ == '__main__':
    new_weights = tf.ones((3, 256, 256, 3), dtype=tf.float32)
    dummy_out = tf.random.normal((3, 1))

    z = tf.split(new_weights, num_or_size_splits=[256//8] * 8, axis=1)
    x_init = tf.keras.layers.Flatten()(z[0])
    x = Linear(units=16*16*16*1)(x_init)
    x = tf.reshape(x, shape=[-1, 16, 16, 16])
    model = SelfAttention(channels=16, sn=False)
    model.build(input_shape=(3, 16, 16, 16))
    # out = model(x)
    # print(out)
    print(model.summary())
    # logdir = "../logs/graph/"
    # if not os.path.exists(logdir):
    #     os.makedirs(logdir)
    #
    # tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, write_graph=False)
    # model.compile(
    #     optimizer='adam',
    #     loss='binary_crossentropy',
    #     metrics=['accuracy']
    # )
    # print("the input before fit: ", x.get_shape())
    # model.fit(
    #     x,
    #     dummy_out,
    #     batch_size=1,
    #     epochs=1,
    #     callbacks=[tensorboard_callback]
    # )

