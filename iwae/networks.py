import numpy as np
import tensorflow as tf
from tensorflow.python.ops import init_ops
from tensorflow.python import debug as tf_debug
import utils
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
tiny_default = 1e-32

def build_dense_layers(input_tensor, hidden_units, activation_function, layer_name):
    input_cur = input_tensor
    layer_iter = 1
    output = None
    for hidden_units_cur in hidden_units:
        output = tf.layers.dense(input_cur,hidden_units_cur,activation_function,
                                 name=layer_name+str(layer_iter))
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
        return (-tf.constant(log2pi,dtype=tf.float32, shape=[samples.shape[0]], name='log2pi') \
               - 0.5*tf.reduce_sum(samples**2, axis=1))


class GaussianStochLayer:
    def __init__(self, mean_layer, sigma_layer):
        self.mean_layer = mean_layer
        self.sigma_layer = sigma_layer

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

        mean = tf.layers.dense(input_tensor, latent_units,
                               activation = tf.sigmoid,
                               bias_initializer=init_ops.constant_initializer(mean_bias),
                               name=layer_name + 'mean')  # Linear activation
        return BernoulliStochLayer(mean)

class IWAE:
    def __init__(self, q_layers, p_layers, q_samples, prior, num_samples):
        self.q_layers = q_layers # a list of the graph nodes pertaining to each stochastic layer in the encoder network
        self.p_layers = p_layers # a list of the graph nodes pertaining to each stochastic layer in the decoder network
        self.q_samples = q_samples # samples from each stochastic layer in the encoder network
        self.prior = prior # assumed prior for the latent variables
        self.num_samples = num_samples

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
            ws_matrix = tf.exp(log_ws_minus_max)
            # ws_matrix = tf.Print(ws_matrix,[ws_matrix],message='ws_matrix')
            lb = log_ws_matrix_max + tf.log(tf.reduce_mean(ws_matrix, axis=0),'lowerbound')
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
            ws_matrix  = tf.exp(log_ws_minus_max)
            #ws_matrix = tf.Print(ws_matrix,[ws_matrix],message='ws_matrix')
            lb = log_ws_matrix_max + tf.log(tf.reduce_mean(ws_matrix, axis=0),'lowerbound')
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
        for hidden_units_cur, latent_units_cur in zip(hidden_units_q, latent_units):
            # build the dense hidden layers for this stochastic unit
            dense = build_dense_layers(input_cur, hidden_units_cur,
                                       activation_function=tf.nn.tanh,
                                       layer_name='q_det_unit_' + str(layer_iter) + '_')

            # build the stochastic layer
            layers_q.append(GaussianStochLayer.build_stochastic_layer(dense,latent_units_cur,
                                                                      layer_name='q_stoch_layer_'+str(layer_iter)+'_'))
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
            dense = build_dense_layers(input_cur, hidden_units_cur,
                                       activation_function=tf.nn.tanh,
                                       layer_name='p_det_unit_' + str(layer_iter) + '_')

            # build the stochastic layer
            layers_p.append(GaussianStochLayer.build_stochastic_layer(dense, latent_units_cur,layer_name='p_stoch_layer_' + str(layer_iter) + '_'))
            layer_iter += 1

        # build the last dense layer for the decoder
        dense = build_dense_layers(rev_samples_q[-1], hidden_units_p[-1],
                                   activation_function=tf.nn.tanh,
                                   layer_name='p_det_unit_' + str(layer_iter) + '_')

        # build the last stochastic layer
        if(data_type == 'binary'):
            layers_p.append(BernoulliStochLayer.build_stochastic_layer(dense, input_tensor.shape[1],
                                                                      layer_name='p_stoch_layer_' + str(layer_iter) + '_',
                                                                      mean_bias=bias))
        elif data_type == 'continuous':
            layers_p.append(GaussianStochLayer.build_stochastic_layer(dense, input_tensor.shape[1],
                                                                  layer_name='p_stoch_layer_' + str(layer_iter) + '_',
                                                                  mean_bias=bias))
        prior = UnitGaussianLayer(layers_q[-1].mean_layer.shape)
        return IWAE(layers_q, layers_p, samples_q, prior, num_samples)


def has_nan(datum, tensor):
  return np.any(np.isnan(tensor))


if __name__ == '__main__':
    print('testing the networks module....')

    import datasets_utils as d_util

    dataset = d_util.Dataset(shuffle=True)
    dataset.scale_down_data()

    #x_train_ex_batch, _ = dataset.get_train_minibatch(100)
    #print(x_train_ex_batch.shape)

    num_samples = 5
    batch_size = 2

    #layers
    latent_units = [50]
    hidden_units_q = [[200,200]]
    hidden_units_p = [[200,200]]

    # Create the model
    x = tf.placeholder(tf.float32, [batch_size, dataset.dim], name='placeholder_x')
    iwae = IWAE.build_network(x, num_samples, latent_units, hidden_units_q, hidden_units_p, dataset.train_bias)
    grads, lowerbound  = iwae.get_gradient()
    lowerbound_real = iwae.get_lowerbound_real()
    samples = iwae.get_generated_samples()

    #lowerbound = tf.Print(lowerbound, [lowerbound])
    #print(grads)
    lr = tf.placeholder(tf.float32)
    train_step = tf.train.AdamOptimizer(learning_rate=lr).apply_gradients(grads)

    #sess = tf.InteractiveSession()
    #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    num_epochs = 1
    shuffle_seed = 123
    num_train_iters = int(dataset.train_num / batch_size)
    lr_float = 1e-2

    #sess = tf.Session()
    with tf.Session() as sess:
        # sess = tf.InteractiveSession()
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        #sess.add_tensor_filter("has_nan", has_nan)
        writer = tf.summary.FileWriter('./graphs', sess.graph)
        tf.global_variables_initializer().run()

        #for iter in range(1000):
        #    batch_xs, _ = dataset.get_train_minibatch(batch_size)
        #    lr_float = 1e-3

        #    print('lowerbound on log likelihood ', iter, ': ',sess.run(lowerbound, feed_dict={x: batch_xs}))
        #    #print(sess.run(lowerbound.shape, feed_dict={x: batch_xs}))
        #    #sess.run(lowerbound, feed_dict={x: batch_xs})
        #    output = sess.run(fetches=[train_step], feed_dict={x: batch_xs, lr: lr_float})


        # training
        num_train_iters = 1
        for epoch in range(num_epochs):
            lr_float = lr_float/10
            permutation = np.random.RandomState(seed=shuffle_seed).permutation(dataset.train_num)
            for index in range(num_train_iters):
                mask = permutation[index * batch_size: (index + 1) * batch_size]
                batch_xs, _ = dataset.get_data_from_mask(mask,'train')
                print('lowerbound on log likelihood ', index, ': ', sess.run(lowerbound, feed_dict={x: batch_xs}))
                output = sess.run(fetches=[train_step], feed_dict={x: batch_xs, lr: lr_float})

        num_val_iters = int(dataset.val_num / batch_size)
        num_val_iters = 1
        lower_bound = 0
        for index in range(num_val_iters):
            num_range = np.arange(dataset.val_num)
            mask = num_range[index * batch_size: (index + 1) * batch_size]
            batch_xs, _ = dataset.get_data_from_mask(mask, 'val')
            lb = sess.run(lowerbound_real, feed_dict={x: batch_xs})
            lower_bound += lb
            print('lowerbound on val log likelihood ', index, ': ', lb)
        #sess.close()
        print('avg lowerbound on val : ', lower_bound/dataset.val_num)

        sample_dict = {}
        examples = dataset.get_n_examplesforeachlabel(2,'test')
        for key in sorted(examples):
            X= examples[key]
            sample_dict[key] = sess.run(samples,feed_dict={x: X})
            print('key is : ', key, '     ', sample_dict[key].shape)

        print(len(sample_dict))
        print(sample_dict[0].shape)
        utils.visualize_labelled_examples(sample_dict, dataset.orig_image_shape)

    writer.close()
