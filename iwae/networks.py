import sys, os #load_save
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import init_ops
from tensorflow.python import debug as tf_debug
import utils

log2pi = 0.5 * np.log(2 * np.pi)
tiny_default = 1e-32

def build_dense_layers(input_tensor, hidden_units, activation_function, layer_name):
    input_cur = input_tensor
    layer_iter = 1
    output = None
    params ={}
    for hidden_units_cur in hidden_units:
        output = tf.layers.dense(input_cur,hidden_units_cur,activation_function,
                                 name=layer_name+str(layer_iter))
        # for weights we automatically use init_ops.glorot_uniform_initializer()
        #print(layer_name+str(layer_iter), ' : ', output.kernel)
        param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, layer_name + str(layer_iter) + '/' + 'kernel')

        print('collection : ', layer_name+str(layer_iter), ': ', param[0])
        print(output)
        msg = 'collection :'+layer_name+str(layer_iter)
        #variable =tf.Print(variable,[variable],message=msg)
        params[layer_name+str(layer_iter)] = param[0]
        input_cur = output
        layer_iter += 1
    return output, params

class UnitGaussianLayer:
    def __init__(self, shape):
        self.shape = shape

    def get_samples(self):
        return tf.random_normal(self.shape)

    def log_likelihood_samples(self, samples):
        '''Given samples as rows of a matrix, returns their log-likelihood under the zero mean unit covariance Gaussian as a vector'''
        return (-tf.constant(log2pi,dtype=tf.float32, shape=[samples.shape[0]], name='log2pi') \
               - 0.5*tf.reduce_sum(samples**2, axis=1))


class GaussianStochLayer:
    def __init__(self, mean_layer, sigma_layer,params):
        self.mean_layer = mean_layer
        self.sigma_layer = sigma_layer
        self.params = params

    def get_samples(self):
        unit_gaussian_samples = tf.random_normal(self.mean_layer.shape)
        return self.sigma_layer * unit_gaussian_samples + self.mean_layer

    def log_likelihood_samples(self, samples, tiny=tiny_default):
        '''Given samples as rows of a matrix, returns their log-likelihood under the zero mean unit covariance Gaussian as a vector'''
        return (-tf.constant(log2pi, dtype=tf.float32, shape=[samples.shape[0]], name='log2pi_gauss') \
                - 0.5 * (tf.reduce_sum(((samples - self.mean_layer)/self.sigma_layer)** 2 + 2*tf.log(self.sigma_layer + tiny,'log_sigma_gauss'), axis=1)))

    @staticmethod
    def build_stochastic_layer(input_tensor, latent_units, layer_name, mean_bias = None):
        if(mean_bias == None):
            mean_bias = 0
        params = {}
        mean = tf.layers.dense(input_tensor, latent_units,
                               bias_initializer=init_ops.constant_initializer(mean_bias), name=layer_name+'mean') # Linear activation
        param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, layer_name+'mean' + '/' + 'kernel')

        print('collection : ', layer_name+'mean', ': ', param[0])
        params[layer_name+'mean'] = param[0]
        sigma = tf.layers.dense(input_tensor, latent_units,
                                activation=tf.exp, name=layer_name+'sigma')
        param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, layer_name + 'mean' + '/' + 'kernel')

        print('collection : ', layer_name + 'sigma', ': ', param[0])
        params[layer_name + 'sigma'] = param[0]
        return GaussianStochLayer(mean,sigma,params)

class BernoulliStochLayer:
    def __init__(self, mean_layer, params):
        self.mean_layer = mean_layer
        self.params = params

    def get_samples(self):
        mean = self.mean_layer
        eps = tf.random_uniform(mean.shape)
        samples = tf.where(eps - mean <= 0, tf.ones(mean.shape), tf.zeros(mean.shape))  # N: Use the uniform distribution to get a bernoulli sample
        return samples

    def log_likelihood_samples(self, samples, tiny=tiny_default):
        '''Given samples as rows of a matrix, returns their log-likelihood under the zero mean unit covariance Gaussian as a vector'''
        log_mean = tf.log(self.mean_layer + tiny,name = 'logBernoullicorrectclass')
        #log_mean = tf.Print(log_mean,[log_mean.shape],message='log_mean', summarize=784)
        individual_ll = samples * log_mean + (1 - samples) * tf.log(1 - self.mean_layer + tiny, name='logBernoulliincorrectclass')
        #individual_ll = tf.Print(individual_ll, [individual_ll], message='individual_ll', summarize=785)
        bernoulli_ll = tf.reduce_sum(individual_ll, axis=1)
        #bernoulli_ll = tf.Print(bernoulli_ll,[bernoulli_ll], message='bernoulli_ll',summarize=21)
        return bernoulli_ll

    @staticmethod
    def build_stochastic_layer(input_tensor, latent_units, layer_name, mean_bias=0):
        params = {}
        mean = tf.layers.dense(input_tensor, latent_units,
                               activation = tf.sigmoid,
                               bias_initializer=init_ops.constant_initializer(mean_bias),
                               name=layer_name + 'mean')  # Linear activation
        param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, layer_name + 'mean' + '/' + 'kernel')

        print('collection : ', layer_name + 'mean', ': ', param[0])
        params[layer_name + 'mean'] = param[0]
        return BernoulliStochLayer(mean, params)

class Network:
    def __init__(self, q_layers, p_layers, q_samples, prior, num_samples, params):
        self.q_layers = q_layers # a list of the graph nodes pertaining to each stochastic layer in the encoder network
        self.p_layers = p_layers # a list of the graph nodes pertaining to each stochastic layer in the decoder network
        self.q_samples = q_samples # samples from each stochastic layer in the encoder network
        self.prior = prior # assumed prior for the latent variables
        self.num_samples = num_samples
        self.params = params

    def get_lowerbound_real(self, model_type = 'iwae'):
        # pass

        total_input_size = int(self.q_samples[0].shape[0])
        print('total input size: ',total_input_size)
        print('num_samples: ', self.num_samples)
        minibatch_size = int(total_input_size/self.num_samples)
        # Calculate the importance weights
        log_ws = tf.zeros(total_input_size)
        for q_layer, q_sample, p_layer, q_sample4p in zip(self.q_layers, self.q_samples[1:],
                                              list(reversed(self.p_layers)), self.q_samples[:-1]):
            p_likelihood = p_layer.log_likelihood_samples(q_sample4p, tiny=0)
            #p_likelihood = tf.Print(p_likelihood,[p_likelihood.shape],message='p_likelihood',summarize=30)
            q_likelihood = q_layer.log_likelihood_samples(q_sample, tiny=0)
            #q_likelihood = tf.Print(q_likelihood,[q_likelihood.shape],message='q_likelihood',summarize=785)

            log_ws += p_likelihood - q_likelihood
        prior_likelihood = self.prior.log_likelihood_samples(self.q_samples[-1])
        #prior_likelihood = tf.Print(prior_likelihood,[prior_likelihood],message='prior_l',summarize=30)
        log_ws += prior_likelihood
        # [logwt1_mbe1, logwt1_mbe2,.... logwt1_mben, logwt2_mbe1,...... logwtk_mben] mbe = minibatch element

        log_ws_matrix =  tf.reshape(log_ws,[self.num_samples, minibatch_size], 'log_ws_matrix')
        #log_ws_matrix = tf.Print(log_ws_matrix,[log_ws_matrix, log_ws_matrix.shape], 'log_ws_matrix', summarize=30)
        # [logwt1_mbe1, logwt1_mb2,.... logwt1_mben,
        # logwt2_mbe1, ...................logwt2_mben,
        # ...... ,
        # logwtk_mbe1, ...................logwtk_mben]
        log_ws_matrix_max = tf.reduce_max(log_ws_matrix, axis=0, name='log_ws_matrix_max')
        log_ws_minus_max = log_ws_matrix - log_ws_matrix_max
        #log_ws_minus_max = tf.Print(log_ws_minus_max, [log_ws_minus_max], 'log_ws_minus_max')
        # N: The above is to ensure that the exponent does not overflow
        if model_type in ['vae', 'VAE']:
            lb = tf.reduce_sum(log_ws)/self.num_samples
        else:
            ws_min_matrix = tf.exp(log_ws_minus_max)
            # ws_matrix = tf.Print(ws_matrix,[ws_matrix],message='ws_matrix')
            lb = log_ws_matrix_max + tf.log(tf.reduce_mean(ws_min_matrix, axis=0),'lowerbound')
            #lb = tf.Print(lb,[lb])
        lowerbound = tf.reduce_sum(lb)
        #lowerbound = tf.Print(lowerbound,[lowerbound],message='lowerbound')
        print(lowerbound)
        return lowerbound

    def get_latent_var_samples(self):
        samples = self.q_layers[-1].get_samples()
        return samples

    def get_generated_samples(self):
        samples = self.p_layers[-1].get_samples()
        return samples

    def get_importance_weighted_indices(self):
        total_input_size = int(self.q_samples[0].shape[0])
        print('total input size: ', total_input_size)
        print('num_samples: ', self.num_samples)
        minibatch_size = int(total_input_size / self.num_samples)
        # Calculate the importance weights
        log_ws = tf.zeros(total_input_size)
        for q_layer, q_sample, p_layer, q_sample4p in zip(self.q_layers, self.q_samples[1:],
                                                          list(reversed(self.p_layers)), self.q_samples[:-1]):
            p_likelihood = p_layer.log_likelihood_samples(q_sample4p)
            # p_likelihood = tf.Print(p_likelihood,[p_likelihood.shape],message='p_likelihood',summarize=30)
            q_likelihood = q_layer.log_likelihood_samples(q_sample)
            # q_likelihood = tf.Print(q_likelihood,[q_likelihood.shape],message='q_likelihood',summarize=785)

            log_ws += p_likelihood - q_likelihood
        prior_likelihood = self.prior.log_likelihood_samples(self.q_samples[-1])
        # prior_likelihood = tf.Print(prior_likelihood,[prior_likelihood],message='prior_l',summarize=30)
        log_ws += prior_likelihood
        # [logwt1_mbe1, logwt1_mbe2,.... logwt1_mben, logwt2_mbe1,...... logwtk_mben] mbe = minibatch element

        log_ws_matrix = tf.reshape(log_ws, [self.num_samples, minibatch_size], 'log_ws_matrix')
        # log_ws_matrix = tf.Print(log_ws_matrix,[log_ws_matrix, log_ws_matrix.shape], 'log_ws_matrix', summarize=30)
        # [logwt1_mbe1, logwt1_mb2,.... logwt1_mben,
        # logwt2_mbe1, ...................logwt2_mben,
        # ...... ,
        # logwtk_mbe1, ...................logwtk_mben]
        log_ws_matrix_max = tf.reduce_max(log_ws_matrix, axis=0, name='log_ws_matrix_max')
        log_ws_minus_max = log_ws_matrix - log_ws_matrix_max
        # log_ws_minus_max = tf.Print(log_ws_minus_max, [log_ws_minus_max], 'log_ws_minus_max')
        # N: The above is to ensure that the exponent does not overflow

        ws = tf.exp(log_ws_minus_max, name='ws')
        # ws = tf.Print(ws,[ws])
        # [wt1_mbe1, wt1_mbe2,........ wtk_mben,
        # wt2_mbe1, ...................wtk_mben,
        # ...... ,
        # wtk_mbe1, ...................wtk_mben]

        ws_normalized = ws / tf.clip_by_value(tf.reduce_sum(ws, axis=0, name='reduce_sum_ws'), 1e-9, np.inf,
                                              name='clip_sum_ws')
        # for categorical dist
        ids = tf.multinomial(tf.log(tf.transpose(ws_normalized)), num_samples=1)[:, 0]
        selection_indices = ids * minibatch_size + np.arange(minibatch_size)

        return selection_indices

    def get_importance_weighted_generated_samples(self):

        selection_indices = self.get_importance_weighted_indices()
        samples = self.p_layers[-1].get_samples()

        return tf.gather(samples,selection_indices)

    def get_importance_weighted_means_for_latents(self):

        selection_indices = self.get_importance_weighted_indices()
        selection_indices = tf.Print(selection_indices,[selection_indices],message='selection_indices',summarize=21)
        last_q_layer = self.q_layers[-1]
        means = last_q_layer.mean_layer
        means = tf.Print(means,[means],message='means')

        print(selection_indices.shape)
        print(means.shape)

        return tf.gather(means,selection_indices)

    def get_gradient(self, model_type = 'iwae'):
        # pass

        total_input_size = int(self.q_samples[0].shape[0])
        print('total input size: ',total_input_size)
        print('num_samples: ', self.num_samples)
        minibatch_size = int(total_input_size/self.num_samples)
        # Calculate the importance weights
        log_ws = tf.zeros(total_input_size)
        for q_layer, q_sample, p_layer, q_sample4p in zip(self.q_layers, self.q_samples[1:],
                                              list(reversed(self.p_layers)), self.q_samples[:-1]):
            p_likelihood = p_layer.log_likelihood_samples(q_sample4p)
            #p_likelihood = tf.Print(p_likelihood,[p_likelihood.shape],message='p_likelihood',summarize=30)
            q_likelihood = q_layer.log_likelihood_samples(q_sample)
            #q_likelihood = tf.Print(q_likelihood,[q_likelihood.shape],message='q_likelihood',summarize=785)

            log_ws += p_likelihood - q_likelihood
        prior_likelihood = self.prior.log_likelihood_samples(self.q_samples[-1])
        #prior_likelihood = tf.Print(prior_likelihood,[prior_likelihood],message='prior_l',summarize=30)
        log_ws += prior_likelihood
        # [logwt1_mbe1, logwt1_mbe2,.... logwt1_mben, logwt2_mbe1,...... logwtk_mben] mbe = minibatch element

        log_ws_matrix =  tf.reshape(log_ws,[self.num_samples, minibatch_size], 'log_ws_matrix')
        #log_ws_matrix = tf.Print(log_ws_matrix,[log_ws_matrix, log_ws_matrix.shape], 'log_ws_matrix', summarize=30)
        # [logwt1_mbe1, logwt1_mb2,.... logwt1_mben,
        # logwt2_mbe1, ...................logwt2_mben,
        # ...... ,
        # logwtk_mbe1, ...................logwtk_mben]
        log_ws_matrix_max = tf.reduce_max(log_ws_matrix, axis=0, name='log_ws_matrix_max')
        log_ws_minus_max = log_ws_matrix - log_ws_matrix_max
        #log_ws_minus_max = tf.Print(log_ws_minus_max, [log_ws_minus_max], 'log_ws_minus_max')
        # N: The above is to ensure that the exponent does not overflow

        ws = tf.exp(log_ws_minus_max, name='ws')
        #ws = tf.Print(ws,[ws])
        # [wt1_mbe1, wt1_mbe2,........ wtk_mben,
        # wt2_mbe1, ...................wtk_mben,
        # ...... ,
        # wtk_mbe1, ...................wtk_mben]

        ws_normalized = ws / tf.clip_by_value(tf.reduce_sum(ws, axis=0, name='reduce_sum_ws'), 1e-9, np.inf, name='clip_sum_ws')
        ws_normalized_vector = tf.reshape(ws_normalized, log_ws.shape)
        # [wsnorm1_mbe1, wsnorm1_mbe2,.....wsnorm1_mben,........wsnormk_mben]

        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        if model_type in ['vae', 'VAE']:
            print('Training a VAE')
            gradients = tf.gradients(tf.reduce_sum(-log_ws)/self.num_samples, params)
            grad = zip(gradients, params)
            lb = tf.reduce_sum(log_ws)/self.num_samples
        else:
            print('Training an IWAE')
            ws_normalized_vector_nograd = tf.stop_gradient(ws_normalized_vector, name='importance_weights_no_grad')
            params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            gradients = tf.gradients(tf.reduce_sum(tf.multiply(-log_ws,ws_normalized_vector_nograd)), params)
            #gradients = tf.Print(gradients,[gradients], message ='gradients',summarize=2000)
            grad = zip(gradients, params)
            ws_min_matrix  = tf.exp(log_ws_minus_max)
            #ws_matrix = tf.Print(ws_matrix,[ws_matrix],message='ws_matrix')
            lb = log_ws_matrix_max + tf.log(tf.reduce_mean(ws_min_matrix, axis=0),'lowerbound')
            #lb = tf.Print(lb,[lb])
        lowerbound = tf.reduce_sum(lb)
        #lowerbound = tf.Print(lowerbound,[lowerbound],message='lowerbound')
        print(lowerbound)
        return grad, lowerbound

    @staticmethod
    def build_network(input_batch, num_samples, latent_units, hidden_units_q, hidden_units_p, bias=None, data_type='binary'):

        input_tensor = tf.tile(input_batch, [num_samples, 1], name='tiled_input')

        # encoder
        layers_q = []
        samples_q = []
        input_cur = input_tensor
        #input_tensor = tf.Print(input_tensor,[input_tensor],message='input_tensor', summarize=785)
        samples_q.append(input_tensor)
        layer_iter = 1
        params = {}
        for hidden_units_cur, latent_units_cur in zip(hidden_units_q, latent_units):
            # build the dense hidden layers for this stochastic unit
            dense, variables = build_dense_layers(input_cur, hidden_units_cur,
                                       activation_function=tf.nn.tanh,
                                       layer_name='q_det_unit_' + str(layer_iter) + '_')

            # add variables to the params dict
            utils.add_to_dict(params, variables)

            # build the stochastic layer
            layer_q = GaussianStochLayer.build_stochastic_layer(dense,latent_units_cur,
                                                                      layer_name='q_stoch_layer_'+str(layer_iter)+'_')
            utils.add_to_dict(params,layer_q.params)
            layers_q.append(layer_q)
            input_cur = layers_q[-1].get_samples()
            #input_cur = tf.Print(input_cur, [input_cur], message='samples for layer '+str(layer_iter), summarize=100)
            samples_q.append(input_cur)
            layer_iter += 1

        # decoder
        layers_p = []
        layer_iter = 1
        rev_samples_q = list(reversed(samples_q))[:-1]
        rev_latent_units = list(reversed(latent_units))[1:]
        for hidden_units_cur, latent_units_cur, input_cur in zip(hidden_units_p[:-1], rev_latent_units, rev_samples_q[:-1]):
            # build the dense hidden layers for this stochastic unit
            dense, variables = build_dense_layers(input_cur, hidden_units_cur,
                                       activation_function=tf.nn.tanh,
                                       layer_name='p_det_unit_' + str(layer_iter) + '_')

            # add variables to the params dict
            utils.add_to_dict(params, variables)

            # build the stochastic layer
            layer_p = GaussianStochLayer.build_stochastic_layer(dense, latent_units_cur,layer_name='p_stoch_layer_' + str(layer_iter) + '_')
            utils.add_to_dict(params, layer_p.params)
            layers_p.append(layer_p)
            layer_iter += 1

        # build the last dense layer for the decoder
        dense, variables = build_dense_layers(rev_samples_q[-1], hidden_units_p[-1],
                                   activation_function=tf.nn.tanh,
                                   layer_name='p_det_unit_' + str(layer_iter) + '_')

        # add variables to the params dict
        utils.add_to_dict(params, variables)

        # build the last stochastic layer
        if(data_type == 'binary'):
            layer_p = BernoulliStochLayer.build_stochastic_layer(dense, input_tensor.shape[1],
                                                                      layer_name='p_stoch_layer_' + str(layer_iter) + '_',
                                                                      mean_bias=bias)
            utils.add_to_dict(params, layer_p.params)
            layers_p.append(layer_p)
        elif data_type == 'continuous':
            layer_p = GaussianStochLayer.build_stochastic_layer(dense, input_tensor.shape[1],
                                                                  layer_name='p_stoch_layer_' + str(layer_iter) + '_',
                                                                  mean_bias=bias)
            utils.add_to_dict(params, layer_p.params)
            layers_p.append(layer_p)
        prior = UnitGaussianLayer(layers_q[-1].mean_layer.shape)
        return Network(layers_q, layers_p, samples_q, prior, num_samples, params)