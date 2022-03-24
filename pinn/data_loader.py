import numpy as np
import tensorflow as tf

from numpy.random import random


class DataLoader():
    '''
    provides a data loader that samples collocation points
    and converts data to tf-tensor format, suitable for network training
    '''
    def __init__(self, config):

        np.random.seed(config['seed'])

        # read system settings
        self.T = config['T']
        self.y0 = config['y0']

        # read data settings
        self.n_col = config['n_col']

    def array2tensor(self, array, exp_dim=True):
        '''
        aux function: converts numpy-array to tf-tensor, suitable for training
        '''
        if exp_dim:
            array = np.expand_dims(array, axis=1)

        return tf.convert_to_tensor(array, dtype=tf.float32)

    def t_line(self, t_delta=0.01):
        '''
        returns an equally-spaced data array for postprocessing
        and visualization of final predictions
        '''
        t_line = np.arange(0, self.T, t_delta)

        return self.array2tensor(t_line)

    def random_collocation(self):
        '''
        returns an uniformly sampled collocation data set
        '''
        t_col = self.T * random(self.n_col)

        return self.array2tensor(t_col)
