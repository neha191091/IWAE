import numpy as np
import matplotlib
#matplotlib.use('gtk')
import matplotlib.pyplot as plt

def visualize_labelled_examples(example_dict,imgshape=(28,28)):
    num_keys = len(example_dict)
    for key in sorted(example_dict):
        X, _ = example_dict[key]
        num_samples = X.shape[0]
        for i in range(num_samples):
            plt_idx = i * num_keys + key + 1
            plt.subplot(num_samples, num_keys, plt_idx)
            img = X[i].reshape(imgshape)
            plt.imshow(img.astype('uint8'), cmap='Greys')
            plt.axis('off')
            if i == 0:
                plt.title(key)
    plt.show()
