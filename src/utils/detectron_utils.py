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

from utils.image_utils import get_true_images


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

def build_predictor_model(threshold=0.2, model_path=None):
    '''
    Returns predictor and model using detectron2 from files stored in the
    PyPSA Africa drive

    Parameters
    ----------
    threshold : float
        detection threshold
    model_path : str
        path to .pth file

    Returns
    ---------
    predictor : detectron2.DefaultPredictor
    model : torch.nn.Module

    ''' 

    print('Building predictor from path:')
    print(model_path)

    ds_path = f'/content/drive/My Drive/PyPSA_Africa_images/datasets/duke_train/data/' 
    json_path = f'/content/drive/My Drive/PyPSA_Africa_images/datasets/duke_train/labels.json'
    ds_name = 'duke'
    
    if ds_name in DatasetCatalog.list():
        DatasetCatalog.remove(ds_name)
        MetadataCatalog.remove(ds_name)
    
    register_coco_instances(ds_name, {}, json_path, ds_path)
    
    frcnn = 'faster_rcnn_R_101_FPN_3x.yaml'
    
    cfg = get_cfg() # Model Config
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/"+frcnn))

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    if model_path is None:
        model_path = os.path.join('/content', 'drive', 'MyDrive', 'PyPSA_Africa_images', 'models', 
                                    '2021-11-29_frcnn_180000_dukeset', 'model_final.pth')
    
    cfg.MODEL.WEIGHTS = model_path

    print('working with path: ', model_path)
    cfg.INPUT.FORMAT = 'BGR'

    predictor = DefaultPredictor(cfg)                                
    model = build_model(cfg)

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(model_path)

    return predictor, model


def eval_predictor(imgs, model_path=None, threshold=0.1):
    '''
    Method to check performance of model on some imgs (only chekc via plotting, does 
    not return precision scores)

    Parameters
    ----------
    imgs : list of np.array 
        images on which inference will run
    model_path : str
        path to model .pth file
    threshold : float
        cutoff for what is considered as an instance
    
    Returns
    ----------
    -

    '''

    print('Evaluating model stored under:')
    print(model_path)

    predictor, model = build_predictor_model(threshold=threshold, model_path=model_path)
    model.eval()

    for img in imgs:

        out = predictor(img)
        # out = predictor(img[:,:,::-1])
        v = Visualizer(img, MetadataCatalog.get('duke'), scale=1.5)
        out = v.draw_instance_predictions(out["instances"].to("cpu"))
        cv2_imshow(out.get_image())

        with torch.no_grad():
            
            img = predictor.aug.get_transform(img).apply_image(img)
            img = torch.as_tensor(img.astype('float32').transpose(2, 0, 1))
            x = [{'image': img, 'width': 256, 'height': 256}]
            pred = predictor.model(x)
            print(pred)

if __name__ == '__main__':
    
    duke_img_dir = '/content/drive/MyDrive/PyPSA_Africa_images/datasets/duke_val/data/'
    duke_imgs = get_true_images(duke_img_dir, 20)
    
    model_path = '/content/drive/MyDrive/PyPSA_Africa_images/notebooks/pypsa-africa/PISA_models_duke/model_final.pth'
    model_path_cycle = '/content/drive/MyDrive/PyPSA_Africa_images/PISA_models/11_01_2022_fake_maxar_train/model_final.pth'
    
    print('The cycle GAN trained model')
    eval_predictor(duke_imgs, model_path=model_path_cycle)
    
    print('The regular trained model')
    eval_predictor(duke_imgs, model_path=model_path)
    
