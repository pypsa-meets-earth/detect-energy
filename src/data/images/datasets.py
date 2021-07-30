import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
from numpy import random as rd


class DummyData(Dataset):
    """
    Creates Dataset to test models on. The class creates
    images with random pixels. A manually chosen share of
    images contains a black circle to be recognized by the
    network. 
    Creates instances upon the __getitem__ call
    """
    def __init__(self, 
                 p=0.5, 
                 N=10, 
                 dim=256,
                 radius=20,
                 color_std=0.02
                 ):
        """
        Arguments:
            p (float \in [0,1]): 
                share of images with circle
                
            N (int): 
                number of examples in dataset
                
            dim (int): 
                dimension of images 
                
            radius (int): 
                radius of circle in pixels
                
            color_std (float): 
                std in color of circle
        """
        self.p = p
        self.N = N
        self.dim = dim
        self.rad = radius
        self.color_std = color_std
    

    def create_instance(self):
        """
        Creates a single image containing a 
        circle with probability self.p
        Arguments:
         -
         
        ----------
        Returns:
            image: (3 x self.dim x self.dim torch tensor)
                the example itself
                
            bbox: None --> if no circle, 
                  1 x 4 torch.tensor([center_x, center_y, heigth, width]) 
                  --> if circle is present
        """
        img = torch.randn(3, self.dim, self.dim) * 2
        bbox = None
        
        if rd.rand() < self.p:
            
            center_x, center_y = rd.randint(self.dim - 2*self.rad, size=2) + self.rad
            color = rd.normal(loc=0.8, scale=self.color_std)
            
            for pos in range(-self.rad+1, self.rad):
                
                x = int(np.sqrt(self.rad**2 - pos**2))
                img[:, center_x-x:center_x+x, center_y+pos] = color
                
                bbox = torch.tensor([center_x, center_y, 
                                     2*self.rad, 2*self.rad])
        
        return img, bbox
    
    
    def show_instance(self):
        """
        calls self.__getitem__ and visualizes the resulting example with bbox
        """
        img, bbox = self.__getitem__(1)
        vis_bbox(img, bbox)
    
    
    def __getitem__(self, idx):
        """
        For storage reasons creates instance on the spot
        """
        return self.create_instance()



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

    
if __name__ == "__main__":
    dataset = DummyData()
    
    for _ in range(5):
        dataset.show_instance()
    
