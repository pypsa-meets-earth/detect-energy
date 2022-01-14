import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import helper
import numpy as np
import os
import cv2
from google.colab.patches import cv2_imshow
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances


def register(name, path):
    '''
    adds a dataset to the detectron2.DatasetCatalog.
    ! Has to be in coco-format !

    Parameters
    -----------
    name : str
        name of the dataset
    path : str
        path to dataset with  path
                                |--data/<images>
                                L--labels.json

    Returns
    ----------
    -

    '''

    print(f'Adding dataset {name} from path \n {path} \n to \
            DatasetCatalog and MetadataCatalog.') 

    if name in DatasetCatalog.list():
        DatasetCatalog.remove(name)
        MetadataCatalog.remove(name)

    ds_path = os.path.join(path, 'data')
    json_path = os.path.join(path, 'labels.json')
    register_coco_instances(name, {}, json_path, ds_path)


def build_predictor(threshold=0.1, frcnn=True, model_path=None):
    '''
    Loads and returns a detectron2.DefaultPredictor with a Faster R-CNN or RetinaNet model

    Parameters
    ----------
    threshold : float
        detection threshold set in the network
    frcnn : bool
        loads frcnn if true, else retinanet
    model_path : str
        path to the model to load

    Returns
    ----------
    predictor : detectron2.DefaultPredictor
        with desired model

    '''

    if frcnn: 
        print('Obtaining faster RCNN from path:')
    else: 
        print('Obtaining RetinaNet from path:')
    print(model_path)

    if model_path is None: 
        print('Warning: Returned model will be untrained, as no model_path was passed!')
    
    if frcnn:
        current = 'faster_rcnn_R_101_FPN_3x.yaml'

    else:
        current = 'retinanet_R_101_FPN_3x.yaml'
    
    cfg = get_cfg() # Model Config
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/"+current))

    if frcnn:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold

    else:
        cfg.MODEL.RETINANET.NUM_CLASSES = 2
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = threshold
    
    cfg.MODEL.WEIGHTS = model_path
    cfg.INPUT.FORMAT = 'BGR'

    return DefaultPredictor(cfg)                                


def eval_predictor(imgs, model_path=None, frcnn=True, threshold=0.1):
    '''
    Exhibits performance of model on some images
    (Also expects images to have width and height of 256 pixels)

    Parameters
    ----------
    imgs : list of height x width np.array
        list of images. On all images inference will be performed
    model_path : str
        path to model weights
    frcnn : bool
        loads frcnn if true, else retinanet
    threshold : float
        detection threshold set in the network
    
    Returns 
    -----------
    -

    '''

    predictor = build_predictor(threshold=threshold, frcnn=frcnn, model_path=model_path)

    for img in imgs:

        out = predictor(img)
        v = Visualizer(img, MetadataCatalog.get('duke'), scale=1.5)
        out = v.draw_instance_predictions(out["instances"].to("cpu"))
        cv2_imshow(out.get_image())

        with torch.no_grad():
            
            img = predictor.aug.get_transform(img).apply_image(img)
            img = torch.as_tensor(img.astype('float32').transpose(2, 0, 1))
            x = [{'image': img, 'width': 256, 'height': 256}]
            pred = predictor.model(x)
            print(f"Prediction: {pred}")


if __name__ == '__main__':
    pass