import gpflow
import tensorflow as tf
import tensorflow.keras as tfk
import numpy as np

float_type = gpflow.settings.tf_float


class DKL(object):
    """ Implementation of Deep Kernel Learning

        Model description:

        f(x) \sim GP(0, k(net(x), net(x'))

        where `net' is a neural network
    """

    def __init__(self, X, Y, kernel, likelihood):
        self.X, self.Y = X, Y
        self.kernel = kernel
        self.likelihood = likelihood
        self.session = gpflow.get_session()
        self.build_net(output_dim=self.kernel.input_dim)

    def init_inducing(self):

        net_x = self.session.run(self.net_x, feed_dict={self.x_ph:self.X})
        from sklearn import cluster
        kmean = cluster.KMeans()
        kmean.fit(net_x)
        Z = kmean.cluster_centers_
        return Z

    def initialize(self, shuffle=True, batch_size=100):

        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.X, self.Y))
        self.train_dataset = self.train_dataset.shuffle(shuffle).batch(batch_size)
        self.train_iterator = self.train_dataset.make_one_shot_iterator()
        self.train_iter_next = self.train_iterator.get_next()

        self.x_ph = tf.placeholder(shape=[None, self.X.shape[1]], dtype=float_type)
        self.y_ph = tf.placeholder(shape=[None, 1], dtype=float_type)
        self.net_x = self.net(self.x_ph)
        with tf.variable_scope('optimizer') as scope:
            self.model = gpflow.models.SVGP(self.net_x,
                                            self.y_ph,
                                            kern=self.kernel,
                                            likelihood=self.likelihood,
                                            Z=self.init_inducing(),
                                            num_data=self.X.shape[0],
                                            num_latent=10
                                            )

            self.opt = tf.train.AdamOptimizer()
        self.opt_init = tf.variables_initializer(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
        )

        self.opt_tensor = self.opt.minimize(self.model.objective)

    def build_net(self, output_dim):

        with tf.variable_scope("neural_network") as scope:
            input = tfk.layers.Input(shape=(self.X.shape[1],), dtype=float_type)
            hidden = tfk.layers.Dense(500, activation='relu', dtype=float_type)(input)
            hidden = tfk.layers.Dense(200, activation='relu', dtype=float_type)(hidden)
            output = tfk.layers.Dense(output_dim)(hidden)
            self.net = tfk.Model(input, output)

        self.net_var_intializer = tf.variables_initializer(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
        )

        self.session.run(self.net_var_intializer)

    def train(self, num_iter=100, print_interval=10):

        self.session.run(self.opt_init)
        for i in range(num_iter):
            x_batch, y_batch = self.session.run(self.train_iter_next)
            feed_dict = {self.x_ph: x_batch, self.y_ph: y_batch}
            self.session.run(self.opt_tensor, feed_dict=feed_dict)
            if (i + 1) % print_interval == 0:
                l_eval = self.session.run(self.model.likelihood)
                print("Iter =  {} \t Likelihood = {}".format(i, l_eval))

    def test(self):
        pass

import common.dataset

# data
dataset = common.dataset.MnistDataset()
dataset.download()
def flatten_mnist(x):
    shape = x.shape
    return np.reshape(x, newshape=[-1, shape[1]*shape[2]])
train, test = dataset.get_numpy()
X_train, y_train = flatten_mnist(train[0]), train[1][:,None]
X_test, y_test = flatten_mnist(test[0]), test[1][:,None]

# build model
net_output_dim = 30
# with tf.variable_scope('gp') as scope:
kernel = gpflow.kernels.RBF(input_dim=net_output_dim, lengthscales=1., variance=1.)
likelihood = gpflow.likelihoods.MultiClass(num_classes=10)

# gp_var_intializers = tf.variables_initializer(
#     tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
# )
model = DKL(X_train, y_train, kernel, likelihood)

model.initialize()
model.train()
