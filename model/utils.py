import tensorflow as tf
tf.random.set_seed(1231213)


# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# Regularization
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
def orthogonal_regularizer(scale):
    """ Defining the Orthogonal regularizer and return the function at
     last to be used in Conv layer as kernel regularizer"""

    def ortho_reg(w):
        """ Reshaping the matrix in to 2D tensor for enforcing orthogonality"""
        shape = w.get_shape().as_list()
        c = shape[-1]

        w = tf.reshape(w, [-1, c])

        """ Declaring a Identity Tensor of appropriate size"""
        identity = tf.eye(c)

        """ Regularizer Wt*W - I """
        w_transpose = tf.transpose(w)
        w_mul = tf.matmul(w_transpose, w)
        reg = tf.subtract(w_mul, identity)

        """Calculating the Loss Obtained"""
        ortho_loss = tf.nn.l2_loss(reg)

        return scale * ortho_loss

    return ortho_reg


def orthogonal_regularizer_fully(scale):
    """ Defining the Orthogonal regularizer and return the function at last to be used in Fully Connected Layer """

    def ortho_reg_fully(w) :
        """ Reshaping the matrix in to 2D tensor for enforcing orthogonality"""
        _, c = w.get_shape().as_list()

        """Declaring a Identity Tensor of appropriate size"""
        identity = tf.eye(c)
        w_transpose = tf.transpose(w)
        w_mul = tf.matmul(w_transpose, w)
        reg = tf.subtract(w_mul, identity)

        """ Calculating the Loss """
        ortho_loss = tf.nn.l2_loss(reg)

        return scale * ortho_loss

    return ortho_reg_fully


# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# Sampling
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+


# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# Normalization
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

def l2_norm(x, eps=1e-12):
    return x / (tf.sqrt(tf.reduce_sum(tf.square(x))) + eps)


def batch_norm(momentum=0.9, eps=1e-3, center=True, scaling=True, is_train=True, name="bn"):
    return tf.keras.layers.BatchNormalization(
        momentum=momentum,
        epsilon=eps,
        center=center,
        scale=scaling,
        trainable=is_train,
        name=name)


def pixel_norm(x, eps=1e-3):
    return x / tf.sqrt(tf.reduce_mean(tf.square(x), axis=[1, 2, 3]) + eps)


class SpectralNorm(tf.keras.layers.Layer):
    def __init__(self, gain=1., iteration=1, **kwargs):
        super(SpectralNorm, self).__init__(**kwargs, dynamic=True)
        self.gain = gain
        self.iteration = iteration
        self.u = None

    def build(self, input_shape):
        self.u = self.add_variable(shape=[1, input_shape[-1]],
                                   initializer=tf.initializers.TruncatedNormal(stddev=self.gain),
                                   trainable=False)
        super(SpectralNorm, self).build(input_shape)

    def call(self, w_in):
        shape = tf.shape(w_in)
        w = tf.reshape(w_in, shape=[-1, shape[-1]])
        u_hat = self.u
        for i in range(self.iteration):
            v_ = tf.matmul(u_hat, tf.transpose(w))
            v_hat = tf.nn.l2_normalize(v_)

            u_ = tf.matmul(v_hat, w)
            u_hat = tf.nn.l2_normalize(u_)

        u_hat = tf.stop_gradient(u_hat)
        v_hat = tf.stop_gradient(v_hat)

        sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
        with tf.control_dependencies([self.u.assign(u_hat)]):
            w_norm = w / sigma
            w_norm = tf.reshape(w_norm, w_in.get_shape())
        return w_norm

    def compute_output_shape(self, input_shape):
        return input_shape


if __name__ == '__main__':
    print("working on resblocks")
    weights = tf.random.normal((1, 3, 3, 1))
    print("Before normalization:\n{}".format(weights))
    normalized = SpectralNorm()(weights)
    print("After normalization:\n {}".format(normalized))
