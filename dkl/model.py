import gpflow
import tensorflow as tf

class DKL(object):

    """ Implementation of Deep Kernel Learning

        Model description:

        f(x) \sim GP(0, k(net(x), net(x'))

        where `net' is a neural network
    """

    def __init__(self, kernel, likelihood):
        self.kernel = kernel
        self.likelihood = likelihood
        self.session = gpflow.get_session()
        self.build_net()

    def initialize(self):
        pass


    def load_data(self):
        pass

    def build_net(self):
        self.net = None
        self.net_var_intializer = None
        self.session.run(self.net_var_intializer)

    def train(self, num_iter=100):
        pass

    def test(self):
        pass



