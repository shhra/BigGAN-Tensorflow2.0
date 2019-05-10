import tensorflow as tf
from model.ops import *


class GeneratorBlock(tf.keras.Model):
    def __init__(self, z, z_dim, ch, c_dim, is_training=True):
        super(GeneratorBlock, self).__init__()
        self.z = z
        self.z_dim = z_dim
        self.ch = ch
        self.c_dim = c_dim
        self.is_training = is_training

        if self.z_dim == 128:
            split_dim = 18
            split_dim_remainder = self.z_dim - (split_dim * 6)
            self.z_split = tf.split(self.z, num_or_size_splits=[split_dim] * 6 + [split_dim_remainder], axis=-1)
        else:
            split_dim = self.z_dim // 7
            split_dim_remainder = self.z_dim - (split_dim * 7)

            if split_dim_remainder == 0:
                self.z_split = tf.split(self.z, num_or_size_splits=[split_dim] * 7, axis=-1)
            else:
                self.z_split = tf.split(self.z, num_or_size_splits=[split_dim] * 6 + [split_dim_remainder], axis=-1)

        self.flat = tf.keras.layers.Flatten()
        self.linear = Linear(units=4 * 4 * self.ch)
        self.res_up_16_to_16 = ResBlockUp(self.z_split[1], channels=self.ch, use_bias=False)
        self.res_up_16_to_8 = ResBlockUp(self.z_split[2], channels=self.ch // 2, use_bias=False)
        self.res_up_8_to_8 = ResBlockUp(self.z_split[3], channels=self.ch // 2, use_bias=False)
        self.res_up_8_to_4 = ResBlockUp(self.z_split[4], channels=self.ch // 4, use_bias=False)
        self.res_up_4_to_2 = ResBlockUp(self.z_split[5], channels=self.ch // 8, use_bias=False)
        self.non_local = SelfAttention(channels=ch // 8, sn=False)
        self.res_up_2_to_1 = ResBlockUp(self.z_split[6], channels=self.ch // 16, use_bias=False)
        self.bn = batch_norm(is_train=is_training)
        self.relu = tf.keras.layers.ReLU()
        self.conv = Conv(filters=self.c_dim, kernel=3, stride=1, pad=1, use_bias=False)
        self.activate = tf.keras.activations.tanh
        self.final_shape = None

    def call(self, inputs):
        x = self.flat(inputs)
        x = self.linear(x)
        x = tf.reshape(x, shape=[-1, 4, 4, self.ch])
        x = self.res_up_16_to_16(x)
        x = self.res_up_16_to_8(x)
        x = self.res_up_8_to_8(x)
        x = self.res_up_8_to_4(x)
        x = self.res_up_4_to_2(x)
        x = self.non_local(x)
        x = self.res_up_2_to_1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)
        print("The output shape of final is ", x.shape)
        x = self.activate(x)
        self.final_shape = x.shape
        return x

    def compute_output_shape(self, input_shape):
        return self.final_shape


class DiscriminatorBlock(tf.keras.Model):
    def __init__(self, ch, use_bias, sn, is_training=True):
        super(DiscriminatorBlock, self).__init__()
        self.ch = ch
        self.use_bias = use_bias
        self.sn = sn
        self.is_training = is_training
        self.res_down_1ch_to_2ch = ResBlockDown(channels=self.ch, use_bias=self.use_bias, sn=self.sn, is_training=self.is_training)
        self.res_down_2ch_to_4ch = ResBlockDown(channels=self.ch*2, use_bias=self.use_bias, sn=self.sn, is_training=self.is_training)
        self.non_local = SelfAttention(channels=ch*2, sn=self.sn, is_training=self.is_training)
        self.res_down_4ch_to_8ch = ResBlockDown(channels=self.ch*4, use_bias=self.use_bias, sn=self.sn, is_training=self.is_training)
        self.res_down_8ch_to_8ch = ResBlockDown(channels=self.ch*4, use_bias=self.use_bias, sn=self.sn, is_training=self.is_training)
        self.res_down_8ch_to_16ch = ResBlockDown(channels=self.ch*8, use_bias=self.use_bias, sn=self.sn, is_training=self.is_training)
        self.res_down_16ch_to_16ch = ResBlockDown(channels=self.ch*8, use_bias=self.use_bias, sn=self.sn, is_training=self.is_training)
        self.res_16ch_to_16ch = ResBlock(channels=self.ch*8, use_bias=self.use_bias, sn=self.sn)
        self.relu = tf.keras.layers.ReLU()
        self.gsp = global_sum_pooling
        # Insert Embedding here << << <<
        self.linear = Linear(units=1, sn=self.sn)

    def call(self, inputs, is_training=True):
        x = self.res_down_1ch_to_2ch(inputs)
        x = self.res_down_2ch_to_4ch(x)
        x = self.non_local(x)
        x = self.res_down_4ch_to_8ch(x)
        x = self.res_down_8ch_to_8ch(x)
        x = self.res_down_8ch_to_16ch(x)
        x = self.res_down_16ch_to_16ch(x)
        x = self.res_16ch_to_16ch(x)
        x = self.relu(x)
        x = self.gsp(x)
        x = self.linear(x)

        return x

    def compute_output_shape(self, input_shape):
        return self.linear.compute_output_shape(input_shape)


class BigGAN256:

    def __init__(self, args):
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.c_dim = 3
        self.sn = True

        """ Generator """
        self.ch = args.ch
        self.z_dim = args.z_dim  # dimension of noise-vector
        # self.gan_type = args.gan_type

        """ Discriminator """
        self.n_critic = args.n_critic
        # self.sn = args.sn
        # self.ld = args.ld
