import os
import tensorflow as tf

class Dataset(object):

    def __init__(self, path="../data", name="default"):
        self.path = path
        self.name = name

    def download(self):
        pass

    def get_numpy(self):
        """
        Just return numpy version of train and test
        :return:
        """
        raise NotImplementedError()

    def get_tf_iterator(self, **kwargs):
        """
        Create tensorflow type iterators
        :return: train iterator and test_iterator
        """
        raise NotImplementedError()

    def get_torch_iterator(self, **kwargs):
        """
        Create pytorch type iterator
        :return:
        """
        raise NotImplementedError()


class MnistDataset(Dataset):

    def __init__(self, path="../data"):
        super().__init__(path, name="MNIST")

    def download(self):
        path = os.path.join(self.path, self.name)
        if not os.path.isdir(path):
            os.mkdir(path)

        self.train, self.test = tf.keras.datasets.mnist.load_data(self.name)

    def get_numpy(self):
        return self.train, self.test

    def get_tf_iterator(self, shuffle=True, batch_size=100):
        train = tf.data.Dataset.from_tensor_slices(self.train).shuffle(shuffle).batch(batch_size)
        test = tf.data.Dataset.from_tensor_slices(self.test).shuffle(shuffle).batch(batch_size)
        return train.make_one_shot_iterator(), test.make_one_shot_iterator()
