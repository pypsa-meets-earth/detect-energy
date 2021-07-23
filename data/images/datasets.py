import torch
from torch.utils.data import Dataset
import numpy as np
from numpy import random as rd

from utils import vis_bbox


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
        calls self.__getitem__ and visualizes the respective
        """
        img, bbox = self.__getitem__(1)
        vis_bbox(img, bbox)
    
    
    def __getitem__(self, idx):
        """
        For storage reasons creates instance on the spot
        """
        return self.create_instance()
    
    
if __name__ == "__main__":
    dataset = DummyData()
    
    for _ in range(5):
        dataset.show_instance()
    