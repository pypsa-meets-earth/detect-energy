import os
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import cv2
from google.colab.patches import cv2_imshow


def get_true_images(dir, num, show=False):
    '''
    loads a list of images that all have a tower in them. dir has to be a directory
    only containing such images

    Parameters
    ----------
    dir : str
        directory of true examples
    num : int
        length of list returned
    show : bool
        if True, some of the examples are shown

    Returns
    ----------
    imgs : list of 3 x height x width np.ndarray
        list of obtained images
    '''

    imgs = os.listdir(dir) 
    imgs = [os.path.join(dir, img) for img in imgs]
    # randomize order
    np.random.shuffle(imgs)

    imgs = [cv2.imread(img) for img in imgs[:num]]

    if show:
        print('here are 10 images we have obtained')
        for img in imgs[:10]:
            _, ax = plt.subplots(1, 1, figsize=(4, 4))
            ax.imshow(img)
            plt.show()
    
    return imgs    