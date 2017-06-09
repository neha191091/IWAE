import numpy as np
import matplotlib
#matplotlib.use('gtk')
import matplotlib.pyplot as plt
import sys, os
import tensorflow as tf

def visualize_labelled_examples(example_dict,imgshape=(28,28), save=False, savepath=None):
    num_keys = len(example_dict)
    for key in sorted(example_dict):
        X = example_dict[key]
        print(X.shape)
        num_samples = X.shape[0]
        for i in range(num_samples):
            plt_idx = i * num_keys + key + 1
            plt.subplot(num_samples, num_keys, plt_idx)
            img = X[i].reshape(imgshape)
            plt.imshow(img.astype('uint8'), cmap='Greys')
            plt.axis('off')
            if i == 0:
                plt.title(key)
    if save:
        plt.savefig(savepath)
    plt.show()

def img_path_name(dataset_name, num_samples, traintype, dataset_type, imagenum=0, extra_string=None):
    folder_name = 'imgs/'
    path = folder_name + dataset_name + '_' + dataset_type + '_' + traintype+'_k%d' % (num_samples)
    if extra_string is not None:
        path += '_' + extra_string
    fname = path + 'imaget%d.png' % imagenum
    return fname

def ckpt_path_name(dataset_name, num_samples, traintype, extra_string=None):
    path = 'ckpts/' + dataset_name + '/'
    folder_name = traintype+'_k%d' % (num_samples)
    if extra_string is not None:
        folder_name += '_' + extra_string

    path = path + folder_name + '/'
    return path

def save_checkpoint(sess, path, checkpoint=1, var_list=None):
    if not os.path.exists(path):
        os.makedirs(path)
        # save model
    fname = path + 'checkpoint%d.ckpt' % checkpoint
    saver = tf.train.Saver(var_list)
    save_path = saver.save(sess, fname)
    print("Model saved in %s" % save_path)


def load_checkpoint(sess, path, checkpoint=1):
    # load model
    fname = path + 'checkpoint%d.ckpt' % checkpoint
    try:
        saver = tf.train.Saver()
        saver.restore(sess, fname)
        print("Model restored from %s" % fname)
    except:
        print("Failed to load model from %s" % fname)