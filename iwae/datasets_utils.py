import tensorflow as tf
import numpy as np

import struct
import os
import scipy.io
import config

class Dataset():
    def __init__(self,num_val=400,type = 'mnist',shuffle = False):
        self.data = {}
        self.orig_image_shape = (0,0)
        self.dim = 0
        self.train_mean = 0
        self.std_dev = 1

        if(type == 'mnist'):
            self.get_mnist_data(num_val)
        else:
            print('Type must be "mnist" ')

    def get_mnist_data(self, num_val, shuffle):

        def load_mnist_images_np(imgs_filename, labels_filename):
            with open(imgs_filename, 'rb') as f:
                f.seek(4)
                nimages, rows, cols = struct.unpack('>iii', f.read(12))
                dim = rows * cols
                if(imgs_filename.find(config.MNIST_TRAIN_DATA) != -1):
                    self.dim = dim
                    self.orig_image_shape = (rows,cols)
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

        self.data['train_imgs'] = train_imgs[:-num_val]
        self.data['train_labels'] = train_labels[:-num_val]
        self.data['val_imgs'] = train_imgs[-num_val:]
        self.data['val_labels'] = train_labels[-num_val:]
        self.data['test_imgs'] = test_imgs[-num_val:]
        self.data['test_labels'] = test_labels[-num_val:]



