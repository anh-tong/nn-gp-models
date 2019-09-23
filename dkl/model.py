import gpflow
import tensorflow as tf

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
        pass

    def initialize(self):
        self.x_ph = tf.placeholder(shape=[None, self.X.shape[1]], dtype=float_type)
        self.y_ph = tf.placeholder(shape=[None, self.Y.shape[1]], dtype=float_type)
        self.net_x = self.net(self.x_ph)
        self.model = gpflow.models.SVGP(self.net_x,
                                      self.y_ph,
                                      kern=self.kernel,
                                      likelihood=self.likelihood,
                                      Z = self.init_inducing())
        self.opt = gpflow.training.AdamOptimizer()
        self.opt_tensor = self.opt.make_optimize_tensor(self.model)

    def build_net(self):
        self.net = None
        self.net_var_intializer = None
        self.session.run(self.net_var_intializer)

    def train(self, num_iter=100, print_interval=10):

        for i in range(num_iter):
            self.session.run(self.opt_tensor)
            if (i + 1) % print_interval == 0:
                l_eval = self.session.run(self.model.likelihood)
                print("Iter =  {} \t Likelihood = {}".format(i, l_eval))


    def test(self):
        pass




