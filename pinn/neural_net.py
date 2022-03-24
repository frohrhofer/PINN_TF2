import tensorflow as tf

from pinn.data_loader import DataLoader
from pinn.loss_functions import Loss
from pinn.callback import CustomCallback


class PhysicsInformedNN(tf.keras.Sequential):
    '''
    provides the basic Physics-Informed Neural Network class
    with hard constraints for initial conditions
    '''
    def __init__(self, config, verbose=False):

        # call parent constructor & build NN
        super().__init__(name='PINN')
        self.build_network(config, verbose)
        # create data loader instance
        self.data_loader = DataLoader(config)
        # create loss instance
        self.loss = Loss(self, config)
        # create callback instance
        self.callback = CustomCallback(config)

        # training settings
        self.n_epochs = config['n_epochs']
        self.learning_rate = config['learning_rate']

        # hard constraint settings
        self.y0 = config['y0']

        print('*** PINN build & initialized ***')

    def build_network(self, config, verbose):
        '''
        builds the basic PINN architecture based on
        a Keras 'Sequential' model
        '''
        # set random seeds
        tf.random.set_seed(config['seed'])

        # layer settings
        n_hidden = config['n_hidden']
        n_neurons = config['n_neurons']
        activation = config['activation']

        # create keraes Sequential model
        self.neural_net = tf.keras.Sequential()
        # build input layer
        self.neural_net.add(tf.keras.layers.InputLayer(input_shape=(1,)))
        # build hidden layers
        for i in range(n_hidden):
            self.neural_net.add(tf.keras.layers.Dense(units=n_neurons,
                                                      activation=activation))
        # build linear output layer
        self.neural_net.add(tf.keras.layers.Dense(units=1,
                                                  activation=None))
        if verbose:
            self.neural_net.summary()

    def call(self, t):
        '''
        overrites the call function of the (outer) PINN network
        used to implement the hard constraints
        '''
        # hyperbolic tangent distance function
        return self.y0 + tf.math.tanh(t) * self.neural_net(t)
        # linear distance function
        # return self.y0 + t * self.neural_net(t)

    def train(self):
        '''
        trains the PINN using Adam and anew sampled collocation points
        at each epoch.
        at the end of training, model is validated and logs are saved
        '''
        # Adam optimizer with default settings for exponential decay rates
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate)

        print("Training started...")
        for epoch in range(self.n_epochs):

            # sample collocation points
            t_col = self.data_loader.random_collocation()
            # perform one train step
            train_logs = self.train_step(t_col)
            # provide logs to callback
            self.callback.write_logs(train_logs, epoch)

        # final model validation/prediction
        log = self.validate()
        self.callback.write_logs(log, self.n_epochs)
        # save logs
        self.callback.save_logs()
        print("Training finished!")

    @tf.function
    def train_step(self, t_col):
        '''
        performs a single SGD training step by minimizing the
        physics loss residuals using MSE
        '''
        # open a GradientTape to record forward/loss pass
        with tf.GradientTape() as tape:
            # get physics loss residuals
            res_F = self.loss.F_residuals(t_col)
            # MSE minimization
            loss_F = tf.reduce_mean(tf.square(res_F))

        # retrieve gradients
        grads = tape.gradient(loss_F, self.neural_net.weights)
        # perform single GD step
        self.optimizer.apply_gradients(zip(grads, self.neural_net.weights))

        # save logs for recording
        train_logs = {'loss_F': loss_F}
        return train_logs

    def validate(self):
        '''
        final prediction and physic loss residuals
        '''
        # equally-spaced points
        t_line = self.data_loader.t_line()
        # get final prediction
        y_pred = self(t_line)
        # get final physics loss residuals
        res_F = self.loss.F_residuals(t_line)

        log = {'t_line': t_line.numpy()[:, 0].tolist(),
               'y_pred': y_pred.numpy()[:, 0].tolist(),
               'res_F': res_F.numpy()[:, 0].tolist()}
        return log
