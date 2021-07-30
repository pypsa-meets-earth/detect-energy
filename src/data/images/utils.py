import numpy as np
import torch
import matplotlib.pyplot as plt



def vis_bbox(img, bbox, label=None, score=None, ax=None):
    """Visualize bounding boxes inside image.
    Args:
        img (~numpy.ndarray): An array of shape :math:`(3, height, width)`.
            This is in RGB format and the range of its value is
            :math:`[0, 255]`.
        bbox (~numpy.ndarray): An array of shape :math:`(R, 4)`, where
            :math:`R` is the number of bounding boxes in the image.
            Each element is organized
            by :math:`(y_{min}, x_{min}, y_{max}, x_{max})` in the second axis.
        label (~numpy.ndarray): An integer array of shape :math:`(R,)`.
            The values correspond to id for label names stored in
            :obj:`label_names`. This is optional.
        score (~numpy.ndarray): A float array of shape :math:`(R,)`.
             Each value indicates how confident the prediction is.
             This is optional.
        label_names (iterable of strings): Name of labels ordered according
            to label ids. If this is :obj:`None`, labels will be skipped.
        ax (matplotlib.axes.Axis): The visualization is displayed on this
            axis. If this is :obj:`None` (default), a new axis is created.
    Returns:
        ~matploblib.axes.Axes:
        Returns the Axes object with the plot for further tweaking.
    """

    if isinstance(img, torch.Tensor):
        img = img.numpy()
        
    if isinstance(img, torch.Tensor):
        bbox = bbox.numpy()

    # Returns newly instantiated matplotlib.axes.Axes object if ax is None
    ax = vis_image(img, ax=ax)
    
    if bbox is None:
        return ax

    # If there is no bounding box to display, visualize the image and exit.
    if len(bbox) == 0:
        return ax

    if len(bbox.shape) == 1:
        bbox = np.expand_dims(bbox, axis=0)
    
    for i, bb in enumerate(bbox):
        
        height = bb[2]
        width = bb[3]
        
        xy = (bb[1]-height//2, bb[0]-width//2)
        
        ax.add_patch(plt.Rectangle(
            xy, width, height, fill=False, edgecolor='red', linewidth=2))
        
        caption = list()

        if label is not None and label_names is not None:
            lb = label[i]
            if not (-1 <= lb < len(label_names)):  # modfy here to add backgroud
                raise ValueError('No corresponding name is given')
            caption.append(label_names[lb])
        if score is not None:
            sc = score[i]
            caption.append('{:.2f}'.format(sc))

        if len(caption) > 0:
            ax.text(bb[1], bb[0],
                    ': '.join(caption),
                    style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0})
    return ax


def vis_image(img, ax=None):
    """Visualize a color image.
    Args:
        img (~numpy.ndarray): An array of shape :math:`(3, height, width)`.
            This is in RGB format and the range of its value is
            :math:`[0, 255]`.
        ax (matplotlib.axes.Axis): The visualization is displayed on this
            axis. If this is :obj:`None` (default), a new axis is created.
    Returns:
        ~matploblib.axes.Axes:
        Returns the Axes object with the plot for further tweaking.
    """

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    # CHW -> HWC
    img = img.transpose((1, 2, 0))
    ax.imshow(img.astype(np.uint8))
    return ax