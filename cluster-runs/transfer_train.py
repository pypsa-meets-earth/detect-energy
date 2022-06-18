import os
# os.chdir('/disk3/fioriti/git/detect_energy')
# from dotenv import find_dotenv, load_dotenv
# load_dotenv(find_dotenv())

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"

import sys
# sys.path.append(os.environ.get('PROJECT_ROOT'))
# data_path = os.environ.get('PROJECT_DATASETS')
# model_out_path = "/disk3/fioriti/git/detect_energy/cluster-runs/models/PISA_Parameter_Tuning"

model_out_path = "/content/drive/MyDrive/PyPSA_Africa_images/models/transfer/australia_matched"


from itertools import product
import json
from attrdict import AttrDict
from sklearn.model_selection import ParameterGrid

from detectron2.data import DatasetCatalog
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances 
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader, DatasetMapper, build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageFilter
from typing import List, Union, Optional
import logging
import copy
import random

import detectron2.data.detection_utils as utils
from detectron2.data.dataset_mapper import DatasetMapper

class GaussianBlur:
    """
    Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709
    https://github.com/facebookresearch/moco/blob/master/moco/loader.py
    """

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def build_strong_augmentation(cfg, is_train):
    """
    Create a list of :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """

    logger = logging.getLogger(__name__)
    augmentation = []
    if is_train:
        
        augmentation.append(
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8)
                )
        augmentation.append(transforms.RandomGrayscale(p=0.2))
        augmentation.append(transforms.RandomApply([GaussianBlur([0.1, 1.5])], p=1))

        datatype_transform = transforms.Compose([transforms.ToTensor(), transforms.ToPILImage(),])
        augmentation.append(datatype_transform)
        logger.info("Augmentations used in training: " + str(augmentation))
    return transforms.Compose(augmentation)


DATASETS_PATH = '/content/drive/MyDrive/PyPSA_Africa_images/datasets'

print("DATASETS_PATH: " + DATASETS_PATH)

def register_all():
    # register used datasets
    ds_names = ['australia']
    modes = ['val']

    for name, mode in product(ds_names, modes):

        ds_name = f'{name}_{mode}'
        json_path = os.path.join(DATASETS_PATH, f'{ds_name}/labels.json')
        ds_path = os.path.join(DATASETS_PATH, f'{ds_name}/data/')

        if ds_name in DatasetCatalog.list():
            DatasetCatalog.remove(ds_name)
            MetadataCatalog.remove(ds_name)

        register_coco_instances(ds_name, {}, json_path, ds_path)

    ds_name = 'transmission_04_train'
    json_path = os.path.join(DATASETS_PATH, f'{ds_name}/labels.json')
    ds_path = os.path.join(DATASETS_PATH, f'{ds_name}/data/')

    if ds_name in DatasetCatalog.list():
        DatasetCatalog.remove(ds_name)
        MetadataCatalog.remove(ds_name)

    register_coco_instances(ds_name, {}, json_path, ds_path)

register_all()


class TuneTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

        if isinstance(cfg.DATASETS.EVAL, str):
            self.eval_datasets = [cfg.DATASETS.EVAL]
        else:
            self.eval_datasets = cfg.DATASETS.EVAL

        # prepare evaluation
        self.eval_loaders = []
        self.evaluators = []
        for dataset in self.eval_datasets:

            loader = build_detection_test_loader(DatasetCatalog.get(dataset), 
                                                 mapper=DatasetMapper(cfg, is_train=False))

            self.eval_loaders.append(loader)
            self.evaluators.append(COCOEvaluator(dataset))


    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable
        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True))



    def after_step(self):
        super().after_step()

        if (self.iter+1) % self.cfg.TEST.INTERVAL == 0:                                   

            for dataset, loader, evaluator in zip(self.eval_datasets, 
                                                  self.eval_loaders,
                                                  self.evaluators):

                results = inference_on_dataset(self.model,
                                              loader,
                                              evaluator)
                with open(
                    os.path.join(
                        self.cfg.OUTPUT_DIR,
                        'eval_'+dataset+'_iter_'+str(self.iter)+'.json'),
                        'w') as out:
                    json.dump(results, out)



def run_parameters(params):
    print(f'Starting run for parameters: {params}')
    params = AttrDict(params)

    cfg = get_cfg()

    # From Detectron2 Model Zoo
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/" + params.model_type))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/" + params.model_type)

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.RETINANET.NUM_CLASSES = 1

    cfg.DATASETS.TRAIN = 'transmission_04_train'
    cfg.DATASETS.TEST = ['australia_val']
    cfg.DATASETS.EVAL = ['australia_val']

    cfg.TEST.INTERVAL = 1000
    cfg.SOLVER.MAX_ITER = 12_000
    cfg.SOLVER.STEPS = (8_000, 10_000)

    # setup current parameters
    # cfg.SOLVER.IMS_PER_BATCH = params['SOLVER.IMS_PER_BATCH']
    # cfg.SOLVER.BASE_LR = params['SOLVER.BASE_LR']
    # cfg.SOLVER.MOMENTUM = params['SOLVER.MOMENTUM']
    # cfg.SOLVER.WEIGHT_DECAY = params['SOLVER.WEIGHT_DECAY']
    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = params['MODEL.ANCHOR_GENERATOR.SIZES']

    # select_model = params.model_type.split(".")[0]

    model_name = f"model"
    # model_name +=f"LR_{cfg.SOLVER.BASE_LR}_"
    # model_name +=f"IMSPERBATCH_{cfg.SOLVER.IMS_PER_BATCH}_"
    # model_name +=f"MOM_{cfg.SOLVER.MOMENTUM}_"
    # model_name +=f"WEIGHTDECAY_{cfg.SOLVER.WEIGHT_DECAY}"
    # model_name +=f"ANCHORS"+ \
    #            str(cfg.MODEL.ANCHOR_GENERATOR.SIZES).replace('[[', '_').replace(']]', '_').replace(',', '_').replace(' ', '')

    cfg.OUTPUT_DIR = os.path.join(model_out_path, model_name)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = TuneTrainer(cfg) 
    trainer.resume_or_load(resume=False)

    trainer.train()


if __name__ == '__main__':

    parameters = {
        'model_type': ['faster_rcnn_R_101_FPN_3x.yaml'],
        'SOLVER.BASE_LR': [1e-3],           # default
        'SOLVER.MOMENTUM': [0.9],           # default
        'SOLVER.IMS_PER_BATCH': [8],
        'SOLVER.WEIGHT_DECAY': [0.0001],    # first one is default
        'MODEL.ANCHOR_GENERATOR.SIZES': [[10, 20, 40, 80, 160]],
        }

    parameter_sweep = list(ParameterGrid(parameters))

    for params in parameter_sweep:
        run_parameters(params)