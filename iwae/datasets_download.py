import gzip
import os
import urllib.request as ulr
import config
import sys

if __name__ == '__main__':

    ### Todo: Decide if we want to get the mnist dataset using examples.tutorials


    ### mnist
    #os.chdir(os.path.dirname(sys.argv[0]))

    mnist_filenames = [config.MNIST_TRAIN_DATA, config.MNIST_TRAIN_LABELS, config.MNIST_TEST_DATA, config.MNIST_TEST_LABELS]
    for filename in mnist_filenames:
        local_filename = os.path.join(config.DATASETS_DIR, "MNIST", filename)
        ulr.urlretrieve("http://yann.lecun.com/exdb/mnist/{}.gz".format(filename), local_filename+'.gz')
        with gzip.open(local_filename+'.gz', 'rb') as f:
            file_content = f.read()
        with open(local_filename, 'wb') as f:
            f.write(file_content)
        os.remove(local_filename+'.gz')


    ### binary mnist
    subdatasets = ['train', 'valid', 'test']
    for subdataset in subdatasets:
        filename = 'binarized_mnist_{}.amat'.format(subdataset)
        url = 'http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_{}.amat'.format(subdataset)
        local_filename = os.path.join(config.DATASETS_DIR, "BinaryMNIST", filename)
        ulr.urlretrieve(url, local_filename)