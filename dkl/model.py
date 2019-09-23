import gpflow
import tensorflow as tf
import tensorflow.keras as tfk

float_type = gpflow.settings.float_type


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
        self.build_net()

    def init_inducing(self):

        net_x = self.session.run(self.net, feed_dict={self.x_ph:self.X})
        from sklearn import cluster
        kmean = cluster.KMeans()
        kmean.fit(net_x)
        Z = kmean.cluster_centers_
        return Z

    def initialize(self, shuffle=True, batch_size=100):

        self.train_dataset = tf.data.Dataset.from_tensor_slices((self.X, self.Y))
        self.train_dataset = self.train_dataset.shuffle(shuffle).batch(batch_size)
        self.train_iterator = self.train_dataset.make_one_shot_iterator()

        self.x_ph = tf.placeholder(shape=[None, self.X.shape[1]], dtype=float_type)
        self.y_ph = tf.placeholder(shape=[None, self.Y.shape[1]], dtype=float_type)
        self.net_x = self.net(self.x_ph)
        self.model = gpflow.models.SVGP(self.net_x,
                                        self.y_ph,
                                        kern=self.kernel,
                                        likelihood=self.likelihood,
                                        Z=self.init_inducing())
        self.opt = gpflow.training.AdamOptimizer()
        self.opt_tensor = self.opt.make_optimize_tensor(self.model)

    def build_net(self):

        with tf.variable_scope("neural_network") as scope:
            hidden = tf.layers.Dense(500, activation='relu')(self.x_ph)
            hidden = tf.layers.Dense(200, activation='relu')(hidden)
            self.net = tf.layers.Dense(30)(hidden)

        self.net_var_intializer = tf.variables_initializer(
            tf.get_variable(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)
        )
        self.session.run(self.net_var_intializer)

    def train(self, num_iter=100, print_interval=10):

        for i in range(num_iter):
            x_batch, y_batch = self.session.run(self.train_iterator)
            feed_dict = {self.x_ph: x_batch, self.y_ph: y_batch}
            self.session.run(self.opt_tensor, feed_dict=feed_dict)
            if (i + 1) % print_interval == 0:
                l_eval = self.session.run(self.model.likelihood)
                print("Iter =  {} \t Likelihood = {}".format(i, l_eval))

    def test(self):
        pass
