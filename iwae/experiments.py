from utils import *
from networks import Network
import numpy as np
import tensorflow as tf
import progressbar
from tensorflow.python import debug as tf_debug

def train(sess, input_placeholder, network, dataset, chkpointpath, batch_size=20, num_epochs = 3280, shuffle_seed = 123, checkpoint=0, save=True, optimizer='adam',learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8, model_type = 'iwae'):

    grads, lowerbound = network.get_gradient(model_type=model_type)

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

    #pbar = progressbar.ProgressBar(maxval=(num_epochs-checkpoint) * num_train_iters).start()
    for epoch in range(checkpoint, num_epochs+1):
        #lr_float = lr_float / 10
        i = np.floor(np.log(epoch*2 + 1)/np.log(3))
        lr_float = learning_rate * round(10. ** (1 - (i - 1) / 7.), 1)
        print('learning rate: ', lr_float, ' and i : ', i)
        permutation = np.random.RandomState(seed=shuffle_seed).permutation(dataset.train_num)
        index = 0
        for index in range(num_train_iters):
            mask = permutation[index * batch_size: (index + 1) * batch_size]
            batch_xs, _ = dataset.get_data_from_mask(mask, 'train')
            print('lowerbound on log likelihood at epoch ',epoch,' : ', index, ': ', sess.run(lowerbound, feed_dict={input_placeholder: batch_xs}))
            sess.run(fetches=[train_step], feed_dict={input_placeholder: batch_xs, lr: lr_float})

        # save model
        if save:
            save_checkpoint(sess, chkpointpath, epoch)

        checkpoint += 1
        #pbar.update((epoch-checkpoint) * num_train_iters + index)
    #pbar.finish()

def get_lowerbound(sess, input_placeholder, network, dataset, chkpointpath, checkpoint=0, batch_size=20, dataset_type='val', model_type='iwae',num_samples = 5, save=False, savedir = None):

    if checkpoint>0:
        load_checkpoint(sess, chkpointpath, checkpoint) #Assume that we already have necessary variables initialised if checkpoint=0

    lowerbound_real = network.get_lowerbound_real(model_type=model_type)
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
        #print('lowerbound on ', dataset_type ,' log likelihood ', index, ': ', lb)
    # sess.close()
    print('avg lowerbound on ', dataset_type ,' : ', lower_bound / dataset_size)
    if(save and savedir != None):
        with open(savedir + dataset_type +"_avg_log_likelihood_ckpt"+str(checkpoint)+".txt", "w") as f:
            f.write(str(lower_bound / dataset_size))

def visualize_samples(sess, input_placeholder, network, dataset, chkpointpath, data_shape = None, sample_type = 'generated', save=False, savepath=None, checkpoint=0, num_samples = 5, batch_size=20, dataset_type='val', examples={}, num_examples = 0):
    if checkpoint>0:
        load_checkpoint(sess, chkpointpath, checkpoint) #Assume that we already have necessary variables initialised if checkpoint=0
    else:
        return
    if sample_type == 'generated':
        samples = network.get_generated_samples()
        skip = num_samples
    elif sample_type == 'latent':
        samples = network.get_latent_var_samples()
        skip = num_samples
    elif sample_type == 'generated_importance_weighted':
        samples = network.get_importance_weighted_generated_samples()
        skip = 1
        savepath = savepath
        print('skip', skip)

    if(num_examples == 0):
        num_examples = batch_size
    sample_dict = {}
    if(len(examples) <= 0):
        examples = dataset.get_n_examplesforeachlabel(batch_size, dataset_type)
    for key in sorted(examples):
        X = examples[key]
        sample_dict[key] = sess.run(samples, feed_dict={input_placeholder: X})[:(num_examples)*skip:skip]
        # print('key is : ', key, '     ', sample_dict[key].shape)
    if data_shape == None:
        data_shape = dataset.orig_image_shape
    visualize_labelled_examples(sample_dict, data_shape, save=save, savepath=savepath)

def visualize_2D_posterior(sess, input_placeholder, network, dataset, chkpointpath, save=False, savedir=None, checkpoint=0, batch_size=20, dataset_type='val', num_samples = 5, importance_weighted=False):
    if checkpoint>0:
        load_checkpoint(sess, chkpointpath, checkpoint) #Assume that we already have necessary variables initialised if checkpoint=0
    else:
        return
    if(importance_weighted):
        means = network.get_importance_weighted_means_for_latents()
    else:
        last_q_layer = network.q_layers[-1]
        means = last_q_layer.mean_layer
    if (dataset_type == 'val'):
        dataset_size = dataset.val_num
    elif (dataset_type == 'train'):
        dataset_size = dataset.train_num
    else:
        dataset_size = dataset.test_num

    num_val_iters = int(dataset_size / batch_size)
    #num_val_iters = 1
    z_mu = None
    batch_ys = None
    for index in range(num_val_iters):
        num_range = np.arange(dataset_size)
        mask = num_range[index * batch_size: (index + 1) * batch_size]
        batch_xs, batch_ys = dataset.get_data_from_mask(mask, dataset_type)
        if(importance_weighted):
            batch_ys_rpt = batch_ys
        else:
            batch_ys_rpt = np.reshape(np.tile(batch_ys,num_samples),(1,-1))
        z_mu = sess.run(means, feed_dict={input_placeholder: batch_xs})
        if index==0:
            batch_ys_all = batch_ys_rpt
            z_mu_all = z_mu
        else:
            batch_ys_all = np.concatenate((batch_ys_all,batch_ys_rpt))
            z_mu_all = np.concatenate((z_mu_all,z_mu))
    print(z_mu_all.shape)
    #print(z_mu_all)
    print(batch_ys_all.shape)
    if z_mu != None and batch_ys != None:
        plt.figure(figsize=(8, 6))
        plt.scatter(z_mu_all[:, 0], z_mu_all[:, 1], c=batch_ys_all)
        plt.colorbar()
        plt.grid()
        if save and savedir != None:
            impwtstr = ''
            if(importance_weighted):
                impwtstr ='imp_weighted'
            plt.savefig(savedir + dataset_type + "2_layer_scatter_plot" + str(checkpoint) + impwtstr + ".png")
        plt.close()

def get_units_variances(sess, input_placeholder, network, dataset, chkpointpath, save=False, savedir=None, checkpoint=0, batch_size=20, dataset_type='val'):

    means = []
    for layer in zip(network.q_layers[1:]):
        mean = layer.mean_layer
        means.append(mean)

    if checkpoint>0:
        load_checkpoint(sess, chkpointpath, checkpoint) #Assume that we already have necessary variables initialised if checkpoint=0
    else:
        return

    last_q_layer = network.q_layers[-1]
    means = last_q_layer.mean_layer
    if (dataset_type == 'val'):
        dataset_size = dataset.val_num
    elif (dataset_type == 'train'):
        dataset_size = dataset.train_num
    else:
        dataset_size = dataset.test_num

    num_val_iters = int(dataset_size / batch_size)
    #num_val_iters = 1
    mean_vals = []
    for index in range(num_val_iters):
        num_range = np.arange(dataset_size)
        mask = num_range[index * batch_size: (index + 1) * batch_size]
        batch_xs, batch_ys = dataset.get_data_from_mask(mask, dataset_type)
        if(index == 0):
            mean_vals = sess.run(means, feed_dict={input_placeholder: batch_xs})
            print('shape', mean_vals.shape)
        else:
            mean_vals = np.concatenate((mean_vals,sess.run(means, feed_dict={input_placeholder: batch_xs})))

    vars_of_means = np.var(mean_vals, axis=0)
    print(mean_vals)

    print('vars_of_means', vars_of_means )
    if (save and savedir != None):
        #with open(savedir + dataset_type + "_avg_log_likelihood_ckpt" + str(checkpoint) + ".txt", "w") as f:
        #    f.write(str(vars_of_means))
        plt.hist(np.log(vars_of_means), bins=20)
        plt.savefig(savedir + dataset_type +  "log_variances" + str(checkpoint) + ".png")
        plt.close()
    return vars_of_means

def get_active_units_for_last_stoch_layer(sess, input_placeholder, network, dataset, chkpointpath, threshold=0.01, save=False, savedir=None, checkpoint=0, batch_size=20, dataset_type='val'):
    vars_of_means = get_units_variances(sess,input_placeholder=input_placeholder,network=network,dataset_type=dataset_type
                                        ,dataset=dataset,checkpoint=checkpoint,chkpointpath=chkpointpath,batch_size=batch_size, save=save, savedir=savedir)


    active_units = np.sum(vars_of_means > threshold)
    if (save and savedir != None):
        with open(savedir + dataset_type + "_active_latent_units_chkpoint" + str(checkpoint) + ".txt", "w") as f:
            f.write(str(active_units))
    return active_units

def debug_has_nan(datum, tensor):
  return np.any(np.isnan(tensor))

if __name__ == '__main__':
    
    #Configure options for running the experiment
    get_scatter_plot=False #Get scatter plot for the tst data
    train_model = False #Train the model
    train_vis = False 	#Visualize training data
    val = True		#Visualize validation data
    test = False	#Visualize test data
    compare = False	#Compare vae and iwae

    import datasets_utils as d_util

    dataset = d_util.Dataset(shuffle=True)
    dataset.scale_down_data()

    #x_train_ex_batch, _ = dataset.get_train_minibatch(100)
    #print(x_train_ex_batch.shape)

    num_samples = 30
    batch_size = 20

    #layers
    latent_units = [2]
    #latent_units = [50]
    hidden_units_q = [[200,200]]
    hidden_units_p = [[200,200]]

    # Create the model
    x = tf.placeholder(tf.float32, [batch_size, dataset.dim], name='placeholder_x')
    iwae = Network.build_network(x, num_samples, latent_units, hidden_units_q, hidden_units_p, dataset.train_bias)
    params = iwae.params


    with tf.Session() as sess:

        # for debugging
        #sess = tf.InteractiveSession()
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        #sess.add_tensor_filter("has_nan", debug_has_nan)

        writer = tf.summary.FileWriter('./graphs', sess.graph)

        # Checkpoints
        dataset_name = 'MNIST'
        traintype = 'iwae'
        #path = ckpt_path_name(dataset_name, num_samples, traintype)
        path = ckpt_path_name(dataset_name, num_samples, traintype, extra_string='latents_2')
        #resultsdir = get_results_dir(dataset_name, num_samples, traintype)
        resultsdir = get_results_dir(dataset_name, num_samples, traintype, extra_string='latents_2')
        latest_checkpoint = 30

        num_more_epochs = 30
        num_epochs = latest_checkpoint + num_more_epochs

        # Check if we are getting the weights
        if latest_checkpoint > 0:
            load_checkpoint(sess, path, latest_checkpoint)
            param_dict = sess.run(params)
            for k in param_dict.keys():
                print(k,' : ',param_dict[k].shape)

        if(get_scatter_plot and latest_checkpoint>0 and latent_units[-1] == 2):
            visualize_2D_posterior(sess,input_placeholder=x,network=iwae,dataset=dataset,chkpointpath=path,checkpoint=latest_checkpoint,
                                   batch_size=batch_size,dataset_type='test',save=True,savedir=resultsdir,num_samples=num_samples,importance_weighted=True)

        #active_units = get_active_units_for_last_stoch_layer(sess,input_placeholder=x,network=iwae,dataset_type='test',dataset=dataset,checkpoint=latest_checkpoint,chkpointpath=path,batch_size=batch_size, save=True, savedir=resultsdir)

        if train_model:
            # training
            train(sess,x,iwae,dataset,path,batch_size,num_epochs=num_epochs,checkpoint=latest_checkpoint,model_type=traintype)

        #visualizations
        sample_type = 'generated'

        if train_vis and latest_checkpoint > 0:
            get_lowerbound(sess, x, iwae, dataset, path, checkpoint=latest_checkpoint, batch_size=batch_size,
                           dataset_type='train', model_type=traintype, save=True, savedir=resultsdir)
            # imgpath = img_path_name(dataset_name, num_samples, traintype, 'val', extra_string='latents')
            # visualize_samples(sess, x, iwae, dataset, path, sample_type='latent', data_shape=(10, 5), save=True,
            #                 savepath=imgpath, checkpoint=latest_checkpoint, batch_size=batch_size, dataset_type='val')
            imgpath = img_path_name(dataset_name, num_samples, traintype, 'train', imagenum=latest_checkpoint, extra_string=sample_type)
            visualize_samples(sess, x, iwae, dataset, path, sample_type=sample_type,
                              data_shape=dataset.orig_image_shape, save=True,
                              savepath=imgpath, checkpoint=latest_checkpoint, batch_size=batch_size, dataset_type='train')

        
        if val and latest_checkpoint>0:
            get_lowerbound(sess,x,iwae,dataset,path,checkpoint=latest_checkpoint,batch_size=batch_size,dataset_type='val', model_type=traintype, save=True, savedir = resultsdir)
            #imgpath = img_path_name(dataset_name, num_samples, traintype, 'val', extra_string='latents')
            #visualize_samples(sess, x, iwae, dataset, path, sample_type='latent', data_shape=(10, 5), save=True,
            #                 savepath=imgpath, checkpoint=latest_checkpoint, batch_size=batch_size, dataset_type='val')
            imgpath = img_path_name(dataset_name, num_samples, traintype, 'val', imagenum=latest_checkpoint,extra_string=sample_type)
            visualize_samples(sess, x, iwae, dataset, path, sample_type=sample_type, data_shape=dataset.orig_image_shape, save=True,
                              savepath=imgpath, checkpoint=latest_checkpoint, batch_size=batch_size, dataset_type='val',num_examples=10)

        
        if test and latest_checkpoint>0:
            get_lowerbound(sess, x, iwae, dataset, path, checkpoint=latest_checkpoint, batch_size=batch_size, dataset_type='test',model_type=traintype, save=True,savedir=resultsdir)
            imgpath = img_path_name(dataset_name, num_samples, traintype, 'test', imagenum=latest_checkpoint,extra_string=sample_type)
            visualize_samples(sess, x, iwae, dataset, path, sample_type=sample_type, data_shape=dataset.orig_image_shape, save=True,
                              savepath=imgpath, checkpoint=latest_checkpoint, batch_size=batch_size, dataset_type='test')

        #examples = dataset.get_n_examplesforeachlabel(batch_size, 'val')
        
        if compare and latest_checkpoint>0:
            imgpath = img_path_name(dataset_name, num_samples, 'orig_image', 'test', imagenum=latest_checkpoint)
            examples = dataset.visualize_nlabelled_examples(batch_size,'val',save=True,savepath=imgpath,num_examples_to_show=10)


            vae_path = ckpt_path_name(dataset_name, num_samples, 'vae')
            imgpath = img_path_name(dataset_name, num_samples, 'vae', 'test', imagenum=latest_checkpoint,extra_string='generated')
            visualize_samples(sess, x, iwae, dataset, path, sample_type='generated', data_shape=dataset.orig_image_shape,
                              save=True, savepath=imgpath, checkpoint=latest_checkpoint, num_samples=num_samples, batch_size=batch_size, dataset_type='test', examples=examples, num_examples=10)

            iwae_path = ckpt_path_name(dataset_name, num_samples, 'iwae')
            imgpath = img_path_name(dataset_name, num_samples, 'iwae', 'test', imagenum=latest_checkpoint,
                                    extra_string='generated')
            visualize_samples(sess, x, iwae, dataset, path, sample_type='generated',
                              data_shape=dataset.orig_image_shape,
                              save=True, savepath=imgpath, checkpoint=latest_checkpoint, num_samples=num_samples,
                              batch_size=batch_size, dataset_type='test', examples=examples, num_examples=10)

            iwae_path = ckpt_path_name(dataset_name, num_samples, 'iwae')
            imgpath = img_path_name(dataset_name, num_samples, 'iwae', 'test', imagenum=latest_checkpoint,extra_string='generated_importance_weighted')
            visualize_samples(sess, x, iwae, dataset, path, sample_type='generated_importance_weighted', data_shape=dataset.orig_image_shape,
                              save=True, savepath=imgpath, checkpoint=latest_checkpoint, num_samples=num_samples, batch_size=batch_size, dataset_type='test', examples=examples, num_examples=10)
    writer.close()
