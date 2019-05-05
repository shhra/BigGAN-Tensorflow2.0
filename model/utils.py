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
        _, _, _, c = w.get_shape().as_list()

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


def batch_norm(x, momentum=0.9, eps=1e-3, center=True, scaling=True, is_train=True, name="bn"):
    return tf.keras.layers.BatchNormalization(
        momentum=momentum,
        epsilon=eps,
        center=center,
        scale=scaling,
        trainable=is_train,
        name=name)(x)


def pixel_norm(x, eps=1e-3):
    return x / tf.sqrt(tf.reduce_mean(tf.square(x), axis=[1, 2, 3]) + eps)


def spectral_norm(w, gain=1., iteration=1):
    w_shape = w.get_shape()
    w = tf.reshape(w, [-1, w_shape[-1]])
    initializer = tf.initializers.TruncatedNormal(stddev=gain)
    u = tf.Variable(initializer(shape=(1, w.get_shape()[-1])),
                    name='u',
                    trainable=False,
                    dtype='float32')
    u_hat = u
    v_hat = None
    for i in range(iteration):
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


if __name__ == '__main__':
    print("working on resblocks")
    weights = tf.random.normal((1, 3, 3, 1))
    print("Before normalization:\n{}".format(weights))
    normalized = batch_norm(weights)
    print("After normalization:\n {}".format(normalized))
