import tensorflow as tf
import numpy as np

import struct
import os
import scipy.io
import config
import utils
import matplotlib
import matplotlib.pyplot as plt

# This file contains all the utilities needed for manipulating and extracting batches
# from the datasets assuming that the datasets have already been downloaded to the
# required locations

class Dataset():
    def __init__(self,num_val=400,type = 'mnist', shuffle=False, shuffle_seed=123):

        # Initializes the class object
        # Parameters:
        # num_val:  number of examples for the validation set
        # type: could be <mnist>,<binmnist> or <>


        self.data = {}
        self.orig_image_shape = (0,0)
        self.dim = 0
        self.train_mean = 0
        self.train_std_dev = 1
        self.train_num = 0
        self.classes_num = 0
        self.scaled_down = False
        self.val_num = num_val
        self.test_num = 0

        ## Get Data
        if(type == 'mnist'):
            train_imgs, train_labels, test_imgs, test_labels = self.get_mnist_data()
            if shuffle:
                permutation = np.random.RandomState(seed=shuffle_seed).permutation(train_imgs.shape[0])
                train_imgs = train_imgs[permutation]
                train_labels = train_labels[permutation]

            self.data['train_imgs'] = train_imgs[:-num_val]
            self.data['train_labels'] = train_labels[:-num_val]
            self.data['val_imgs'] = train_imgs[-num_val:]
            self.data['val_labels'] = train_labels[-num_val:]
            self.data['test_imgs'] = test_imgs
            self.data['test_labels'] = test_labels
            self.classes_num = 10
            self.train_num -= self.val_num
            self.test_num = self.data['test_imgs'].shape[0]
        else:
            print('Type must be "mnist" ')
            return

        ## Get mean and standard deviation
        self.train_mean = np.mean(self.data['train_imgs'], axis=0)
        self.train_std_dev = np.std(self.data['train_imgs'],axis=0, keepdims=True) + 1e-7
        self.train_bias = -np.log(1./np.clip(self.train_mean, 0.001, 0.999)-1.)

    def get_mnist_data(self):

        def load_mnist_images_np(imgs_filename, labels_filename):
            with open(imgs_filename, 'rb') as f:
                f.seek(4)
                nimages, rows, cols = struct.unpack('>iii', f.read(12))
                dim = rows * cols
                if(imgs_filename.find(config.MNIST_TRAIN_DATA) != -1):
                    self.dim = dim
                    self.orig_image_shape = (rows,cols)
                    self.train_num = nimages
                images = np.fromfile(f, dtype=np.dtype(np.ubyte)).reshape((nimages, dim))

            with open(labels_filename, 'rb') as f:
                f.seek(4)
                nlabels = struct.unpack('>i', f.read(4))
                labels = np.fromfile(f, dtype=np.dtype(np.ubyte))

            return images, labels

        train_imgs_path = os.path.join(config.DATASETS_DIR, 'MNIST', config.MNIST_TRAIN_DATA)
        train_labels_path = os.path.join(config.DATASETS_DIR, 'MNIST', config.MNIST_TRAIN_LABELS)
        test_imgs_path = os.path.join(config.DATASETS_DIR, 'MNIST', config.MNIST_TEST_DATA)
        test_labels_path = os.path.join(config.DATASETS_DIR, 'MNIST', config.MNIST_TEST_LABELS)

        train_imgs, train_labels = load_mnist_images_np(train_imgs_path, train_labels_path)
        test_imgs, test_labels = load_mnist_images_np(test_imgs_path, test_labels_path)

        return train_imgs, train_labels, test_imgs, test_labels

    def standardize_data(self):
        self.data['train_imgs'] = (self.data['train_imgs'] - self.train_mean) / self.train_std_dev
        self.data['val_imgs'] = (self.data['val_imgs'] - self.train_mean) / self.train_std_dev
        self.data['test_imgs'] = (self.data['test_imgs'] - self.train_mean) / self.train_std_dev

    def scale_down_data(self):
        # In the original paper every value in the image matrix lies between 0 and 1
        self.data['train_imgs'] = np.asarray((self.data['train_imgs']/255.0),dtype=float)
        self.data['val_imgs'] = np.asarray((self.data['val_imgs']/255.0),dtype=float)
        self.data['test_imgs'] = np.asarray((self.data['test_imgs']/255.0),dtype=float)
        self.scaled_down = True

    def get_train_minibatch(self, minibatch_size, replace = True, sample_from_first_n=None):
        if(sample_from_first_n == None):
            sample_from_first_n = self.train_num
        mask = np.random.choice(sample_from_first_n, minibatch_size, replace)
        batch_imgs = self.data['train_imgs'][mask]
        batch_labels = self.data['train_labels'][mask]
        return batch_imgs, batch_labels

    def get_data_from_mask(self, mask, subdataset='train', **kwargs):
        set_imgs = subdataset+'_imgs'
        set_labels = subdataset+'_labels'
        return self.data[set_imgs][mask], self.data[set_labels][mask]

    def get_n_examplesforeachlabel(self, num_examples, set='train'):
        key_imgs = set + '_imgs'
        key_labels = set + '_labels'
        example_dict = {}
        if(len(self.data[key_labels]) <= 0):
            print('No labels present!')
            return None
        #print(self.classes_num)
        for y_hat in range(self.classes_num):
            y = self.data[key_labels]
            #print(y.shape)
            idxs = np.flatnonzero(y == y_hat)
            idxs = np.random.choice(idxs, num_examples, replace=False)
            example_dict[y_hat] = (self.data[key_imgs][idxs])
        return example_dict

    def visualize_nlabelled_examples(self, num_examples=7, set='train'):
        example_dict = self.get_n_examplesforeachlabel(num_examples, set)
        if(self.scaled_down == True):
            for key in sorted(example_dict):
                X = example_dict[key]
                num_samples = X.shape[0]
                for i in range(num_samples):
                    example_dict[key][i] *= 255
        utils.visualize_labelled_examples(example_dict, self.orig_image_shape)


    #def transform_labels2onehot():

if __name__ == '__main__':

    ## Test the module datasets_utils

    dataset = Dataset(shuffle=True)
    # dataset shapes
    print(dataset.data['train_imgs'].shape)
    print(dataset.data['train_labels'].shape)
    print(dataset.data['val_imgs'].shape)
    print(dataset.data['val_labels'].shape)
    print(dataset.data['test_imgs'].shape)
    print(dataset.data['test_labels'].shape)

    # dataset values
    print('dataset values')
    print(dataset.dim)
    print(dataset.train_num)
    print(dataset.classes_num)
    print(dataset.orig_image_shape)
    print(dataset.train_std_dev)
    print(dataset.train_mean.shape)
    print(dataset.train_std_dev.shape)
    print(dataset.train_bias.shape)


    #visualization
    #matplotlib.use('Agg')
    #plt.subplot(2,1,1)
    example_dict = dataset.get_n_examplesforeachlabel(10)
    utils.visualize_labelled_examples(example_dict, dataset.orig_image_shape)

    # standardization
    #dataset.standardize_data()
    #example_dict = dataset.get_n_examplesforeachlabel(2)
    #utils.visualize_labelled_examples(example_dict, dataset.orig_image_shape)

    # scaling
    #plt.subplot(2,1,2)
    dataset.scale_down_data()
    #example_dict = dataset.get_n_examplesforeachlabel(2)
    dataset.visualize_nlabelled_examples(10)

    #plt.show()