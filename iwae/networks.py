import numpy as np
import tensorflow as tf
from tensorflow.python.ops import init_ops

#np.random.seed(0)
#tf.set_random_seed(0)

#def init_weights(input_size, output_size, constant=1.0, seed=123):
#    """ Glorot and Bengio, 2010's initialization of network weights"""
#    scale = constant*np.sqrt(6.0/(input_size + output_size))
#    if output_size > 0:
#        return tf.random_uniform((input_size, output_size),
#                             minval=-scale, maxval=scale,
#                             dtype=tf.float32, seed=seed)
#    else:
#        return tf.random_uniform([input_size],
#                             minval=-scale, maxval=scale,
#                             dtype=tf.float32, seed=seed)

log2pi = 0.5 * np.log(2 * np.pi)

def build_dense_layers(input_tensor, hidden_units, activation_function, layer_name):
    input_cur = input_tensor
    layer_iter = 1
    output = None
    for hidden_units_cur in hidden_units:
        output = tf.layers.dense(input_cur,hidden_units_cur,activation_function,
                                 name=layer_name+layer_iter)
        # for weights we automatically use init_ops.glorot_uniform_initializer()
        input_cur = output
        layer_iter += 1
    return output

class UnitGaussianLayer:
    def __init__(self, shape):
        self.shape = shape

    def get_samples(self):
        return tf.random_normal(self.shape)

    def log_likelihood_samples(self, samples):
        '''Given samples as rows of a matrix, returns their log-likelihood under the zero mean unit covariance Gaussian as a vector'''
        return (-log2pi * tf.constant(log2pi,dtype=float, shape=samples.shape[1]) \
               - 0.5*tf.reduce_sum(samples**2, axis=1))


class GaussianStochLayer:
    def __init__(self, mean_layer, sigma_layer):
        self.mean_layer = mean_layer
        self.sigma_layer = sigma_layer

    def get_samples(self):
        unit_gaussian_samples = tf.random_normal(self.mean_layer.shape)
        return self.sigma_layer * unit_gaussian_samples + self.mean_layer

    def log_likelihood_samples(self, samples):
        '''Given samples as rows of a matrix, returns their log-likelihood under the zero mean unit covariance Gaussian as a vector'''
        return (-log2pi * tf.constant(log2pi, dtype=float, shape=samples.shape[1]) \
                - 0.5 * (tf.reduce_sum(((samples - self.mean_layer)/self.sigma_layer) \
                                       ** 2 + 2*tf.log(self.sigma_layer), axis=1)))

    @staticmethod
    def build_stochastic_layer(input_tensor, latent_units, layer_name, mean_bias = None):
        if(mean_bias == None):
            mean_bias = 0

        mean = tf.layers.dense(input_tensor, latent_units,
                               bias_initializer=init_ops.constant_initializer(mean_bias), name=layer_name+'mean') # Linear activation
        sigma = tf.layers.dense(input_tensor, latent_units,
                                activation=tf.exp, name=layer_name+'sigma')
        return GaussianStochLayer(mean,sigma)

class BernoulliStochLayer:
    def __init__(self, mean_layer):
        self.mean_layer = mean_layer

    def get_samples(self):
        mean = self.mean_layer
        eps = tf.random_uniform(mean.shape)
        samples = tf.where(eps - mean.shape <= 0, tf.ones(mean.shape), tf.zeros(mean.shape))  # N: Use the uniform distribution to get a bernoulli sample
        return samples

    def log_likelihood_samples(self, samples):
        '''Given samples as rows of a matrix, returns their log-likelihood under the zero mean unit covariance Gaussian as a vector'''
        return tf.reduce_sum(samples * tf.log(self.mean_layer) + (1 - samples) * tf.log(1 - self.mean_layer), axis=1)

    @staticmethod
    def build_stochastic_layer(input_tensor, latent_units, layer_name, mean_bias=None):
        if (mean_bias == None):
            mean_bias = 0

        mean = tf.layers.dense(input_tensor, latent_units,
                               bias_initializer=init_ops.constant_initializer(mean_bias),
                               name=layer_name + 'mean')  # Linear activation
        return BernoulliStochLayer(mean)

class IWAE:
    def __init__(self, q_layers, p_layers, q_samples, prior):
        self.q_layers = q_layers # a list of the graph nodes pertaining to each stochastic layer in the encoder network
        self.p_layers = p_layers # a list of the graph nodes pertaining to each stochastic layer in the decoder network
        self.q_samples = q_samples # samples from each stochastic layer in the encoder network
        self.prior = prior # assumed prior for the latent variables

    def error(self):
        pass
    
    @staticmethod
    def build_network(input_tensor, latent_units, hidden_units_q, hidden_units_p, bias=None, data_type='binary'):

        # encoder
        layers_q = []
        samples_q = []
        input_cur = input_tensor
        samples_q.append(input_tensor)
        layer_iter = 1
        for hidden_units_cur, latent_units_cur in zip(hidden_units_q, latent_units):
            # build the dense hidden layers for this stochastic unit
            dense = build_dense_layers(input_cur, hidden_units_cur,
                                       activation_function=tf.nn.tanh,
                                       layer_name='q_det_unit_' + layer_iter + '_')

            # build the stochastic layer
            layers_q.append(GaussianStochLayer.build_stochastic_layer(dense,latent_units,
                                                                      layer_name='q_stoch_layer_'+layer_iter+'_'))
            input_cur = layers_q[-1].get_samples()
            samples_q.append(input_cur)
            layer_iter += 1

        # decoder
        layers_p = []
        layer_iter = 1
        rev_samples_q = list(reversed(samples_q))[:-1]
        rev_latent_units = list(reversed(latent_units))[1:]
        for hidden_units_cur, latent_units_cur, input_cur in zip(hidden_units_p[:-1], rev_latent_units[:-1], rev_samples_q[:-1]):
            # build the dense hidden layers for this stochastic unit
            dense = build_dense_layers(input_cur, hidden_units_cur,
                                       activation_function=tf.nn.tanh,
                                       layer_name='p_det_unit_' + layer_iter + '_')

            # build the stochastic layer
            layers_p.append(GaussianStochLayer.build_stochastic_layer(dense, latent_units,layer_name='p_stoch_layer_' + layer_iter + '_'))
            layer_iter += 1

        # build the last dense layer for the decoder
        dense = build_dense_layers(rev_samples_q[-1], hidden_units_p[-1],
                                   activation_function=tf.nn.tanh,
                                   layer_name='p_det_unit_' + layer_iter + '_')

        # build the last stochastic layer
        if(data_type == 'binary'):
            layers_p.append(BernoulliStochLayer.build_stochastic_layer(dense, rev_latent_units[-1],
                                                                      layer_name='p_stoch_layer_' + layer_iter + '_',
                                                                      mean_bias=bias))
        elif data_type == 'continuous':
            layers_p.append(GaussianStochLayer.build_stochastic_layer(dense, rev_latent_units[-1],
                                                                  layer_name='p_stoch_layer_' + layer_iter + '_',
                                                                  mean_bias=bias))
        prior = UnitGaussianLayer()
        return IWAE(layers_q, layers_p, samples_q, prior)



if __name__ == '__main__':
    print('testing the networks module....')