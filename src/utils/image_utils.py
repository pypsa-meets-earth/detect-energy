import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from PIL import Image
import numpy as np
import cv2

if 'COLAB_GPU' in os.environ:
    from google.colab.patches import cv2_imshow
else:
    from cv2 import imshow as cv2_imshow


__all__ = [
    'get_true_images',
    'downsample',
]


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



def downsample(img, target_size=(256,256), quiet=True):
    '''
    Downsamples img to target_size

    Args:
        img(np.array): original array
        target_size(Tuple[int, int]): desired dimension
    '''

    assert type(img).__module__ == np.__name__
    nr, nc, l = img.shape
    tr, tc = target_size
    shrink_factor = min(tr/nr, tc/nc) # pick the smaller shrink factor

    if not quiet:
        print(f'original width {nc}, height: {nr}')
    img_pil = Image.fromarray(img)
    img_pil = img_pil.resize((round(nc*shrink_factor) ,round(nr*shrink_factor)))
    img_resized = np.array(img_pil)

    nr_, nc_, l_ = img_resized.shape
    if not quiet:
        print(f'resized width {nc_}, height: {nr_} with factor {shrink_factor}')

    return img_resized
