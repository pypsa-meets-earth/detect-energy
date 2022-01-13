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

def build_predictor(threshold=0.2, frcnn=True, model_path=None):

    print('In build predictor:')
    print(model_path)

    ds_path = f'/content/drive/My Drive/PyPSA_Africa_images/datasets/duke_train/data/' 
    json_path = f'/content/drive/My Drive/PyPSA_Africa_images/datasets/duke_train/labels.json'
    ds_name = 'duke'
    
    if ds_name in DatasetCatalog.list():
        DatasetCatalog.remove(ds_name)
        MetadataCatalog.remove(ds_name)
    
    register_coco_instances(ds_name, {}, json_path, ds_path)
    
    retina_net = 'retinanet_R_101_FPN_3x.yaml'
    frcnn = 'faster_rcnn_R_101_FPN_3x.yaml'
    
    if frcnn:
        current = frcnn
    else:
        current = retina_net
    
    cfg = get_cfg() # Model Config
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/"+current))

    if frcnn:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2   # for R-CNN Models
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold  # for R-CNN Models
        if model_path is None:
            model_path = os.path.join('/content', 'drive', 'MyDrive', 'PyPSA_Africa_images', 'models', 
                                        '2021-11-29_frcnn_180000_dukeset', 'model_final.pth')

    else:
        cfg.MODEL.RETINANET.NUM_CLASSES = 2   # for RetinaNet  
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = threshold  # for Retinanet
        model_name = '2021-10-18_retina101_100000_dukeset/model_final.pth'
        if not model_path is None:
            model_path = '/content/drive/MyDrive/PyPSA_Africa_images/models/'+model_name
    
    cfg.MODEL.WEIGHTS = model_path


    print('working with path: ', model_path)
    cfg.INPUT.FORMAT = 'BGR'

    predictor = DefaultPredictor(cfg)                                
    model = build_model(cfg)

    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(model_path)

    return predictor, model


def eval_predictor(imgs, model_path=None, threshold=0.1):

    print('in eval pred: ')
    print(model_path)

    predictor, model = build_predictor(threshold=threshold, frcnn=True, model_path=model_path)
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
    pass