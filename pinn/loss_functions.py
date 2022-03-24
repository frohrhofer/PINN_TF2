import tensorflow as tf


class Loss():
    '''
    provides the physics loss function class
    '''
    def __init__(self, pinn, config):

        # save neural network (weights are updated during training)
        self.pinn = pinn

    def F_residuals(self, t_col):
        '''
        determines physics loss residuals of the differential equation
        at the collocation points
        '''
        # the tf-GradientTape function is used to retreive network derivatives
        with tf.GradientTape() as tape:
            tape.watch(t_col)
            y = self.pinn(t_col)
        y_t = tape.gradient(y, t_col)

        return y_t - y + y**3
