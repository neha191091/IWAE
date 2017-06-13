from utils import *
from networks import Network
import numpy as np
import tensorflow as tf
import progressbar
from tensorflow.python import debug as tf_debug

def train(sess, input_placeholder, network, dataset, chkpointpath, batch_size=20, num_epochs = 3280, shuffle_seed = 123, checkpoint=0, save=True, optimizer='adam',learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):

    grads, lowerbound = network.get_gradient()

    lr = tf.placeholder(tf.float32)
    if(optimizer == 'adam'):
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1, beta2=beta2, epsilon=epsilon)

    train_step = optimizer.apply_gradients(grads)
    learning_rate = learning_rate/10
    lr_float = learning_rate

    # initialize variables
    #checkpoint = 8
    #load = True

    if checkpoint>0:
        load_checkpoint(sess, chkpointpath, checkpoint)
    else:
        tf.global_variables_initializer().run()

    checkpoint += 1
    num_train_iters = int(dataset.train_num / batch_size)
    # num_train_iters = 1

    pbar = progressbar.ProgressBar(maxval=(num_epochs-checkpoint) * num_train_iters).start()
    for epoch in range(checkpoint, num_epochs+1):
        #lr_float = lr_float / 10
        i = np.floor(np.log(epoch*2 + 1)/np.log(3))
        lr_float = learning_rate * round(10. ** (1 - (i - 1) / 7.), 1)
        permutation = np.random.RandomState(seed=shuffle_seed).permutation(dataset.train_num)
        index = 0
        for index in range(num_train_iters):
            mask = permutation[index * batch_size: (index + 1) * batch_size]
            batch_xs, _ = dataset.get_data_from_mask(mask, 'train')
            print('lowerbound on log likelihood ', index, ': ', sess.run(lowerbound, feed_dict={input_placeholder: batch_xs}))
            sess.run(fetches=[train_step], feed_dict={input_placeholder: batch_xs, lr: lr_float})

        # save model
        if save:
            save_checkpoint(sess, chkpointpath, epoch)

        checkpoint += 1
        pbar.update((epoch-checkpoint) * num_train_iters + index)
    pbar.finish()

def get_lowerbound(sess, input_placeholder, network, dataset, chkpointpath, checkpoint=0, batch_size=20, dataset_type='val'):

    if checkpoint>0:
        load_checkpoint(sess, chkpointpath, checkpoint) #Assume that we already have necessary variables initialised if checkpoint=0

    lowerbound_real = network.get_lowerbound_real()
    if(dataset_type == 'val'):
        dataset_size = dataset.val_num
    elif(dataset_type == 'train'):
        dataset_size = dataset.train_num
    else:
        dataset_size = dataset.test_num

    num_val_iters = int(dataset_size / batch_size)
    # num_val_iters = 1
    lower_bound = 0
    for index in range(num_val_iters):
        num_range = np.arange(dataset_size)
        mask = num_range[index * batch_size: (index + 1) * batch_size]
        batch_xs, _ = dataset.get_data_from_mask(mask, dataset_type)
        lb = sess.run(lowerbound_real, feed_dict={input_placeholder: batch_xs})
        lower_bound += lb
        print('lowerbound on val log likelihood ', index, ': ', lb)
    # sess.close()
    print('avg lowerbound on val : ', lower_bound / dataset_size)

def visualize_samples(sess, input_placeholder, network, dataset, chkpointpath, data_shape = None, sample_type = 'generated', save=False, savepath=None, checkpoint=0, batch_size=20, dataset_type='val'):
    if checkpoint>0:
        load_checkpoint(sess, chkpointpath, checkpoint) #Assume that we already have necessary variables initialised if checkpoint=0
    if sample_type == 'generated':
        samples = network.get_generated_samples()
    else:
        samples = network.get_latent_var_samples()
    sample_dict = {}
    examples = dataset.get_n_examplesforeachlabel(batch_size, dataset_type)
    for key in sorted(examples):
        X = examples[key]
        sample_dict[key] = sess.run(samples, feed_dict={input_placeholder: X})[::batch_size]
        # print('key is : ', key, '     ', sample_dict[key].shape)
    if data_shape == None:
        data_shape = dataset.orig_image_shape
    visualize_labelled_examples(sample_dict, data_shape, save=save, savepath=savepath)

def debug_has_nan(datum, tensor):
  return np.any(np.isnan(tensor))

if __name__ == '__main__':
    print('testing the networks module....')

    import datasets_utils as d_util

    dataset = d_util.Dataset(shuffle=True)
    dataset.scale_down_data()

    #x_train_ex_batch, _ = dataset.get_train_minibatch(100)
    #print(x_train_ex_batch.shape)

    num_samples = 5
    batch_size = 20

    #layers
    latent_units = [50]
    hidden_units_q = [[200,200]]
    hidden_units_p = [[200,200]]

    # Create the model
    x = tf.placeholder(tf.float32, [batch_size, dataset.dim], name='placeholder_x')
    iwae = Network.build_network(x, num_samples, latent_units, hidden_units_q, hidden_units_p, dataset.train_bias)

    # Checkpoints
    dataset_name = 'MNIST'
    traintype = 'iwae'
    path = ckpt_path_name(dataset_name, num_samples, traintype)

    with tf.Session() as sess:

        # for debugging
        #sess = tf.InteractiveSession()
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        #sess.add_tensor_filter("has_nan", debug_has_nan)

        writer = tf.summary.FileWriter('./graphs', sess.graph)

        latest_checkpoint = 9
        num_more_epochs = 1
        num_epochs = latest_checkpoint + num_more_epochs

        train_model = False
        if train_model:
            # training
            train(sess,x,iwae,dataset,path,batch_size,num_epochs=num_epochs,checkpoint=latest_checkpoint)

        val = True
        if val:
            get_lowerbound(sess,x,iwae,dataset,path,checkpoint=latest_checkpoint,batch_size=batch_size,dataset_type='val')
            #imgpath = img_path_name(dataset_name, num_samples, traintype, 'val', extra_string='latents')
            #visualize_samples(sess, x, iwae, dataset, path, sample_type='latent', data_shape=(10, 5), save=True,
            #                 savepath=imgpath, checkpoint=latest_checkpoint, batch_size=batch_size, dataset_type='val')
            imgpath = img_path_name(dataset_name, num_samples, traintype, 'val')
            visualize_samples(sess, x, iwae, dataset, path, sample_type='generated', data_shape=dataset.orig_image_shape, save=True,
                              savepath=imgpath, checkpoint=latest_checkpoint, batch_size=batch_size, dataset_type='val')

        test = False
        if test:
            get_lowerbound(sess, x, iwae, dataset, path, checkpoint=8, batch_size=batch_size, dataset_type='test')
            imgpath = img_path_name(dataset_name, num_samples, traintype, 'test')
            visualize_samples(sess, x, iwae, dataset, path, sample_type='generated', data_shape=dataset.orig_image_shape, save=True,
                              savepath=imgpath, checkpoint=latest_checkpoint, batch_size=batch_size, dataset_type='test')

    writer.close()
