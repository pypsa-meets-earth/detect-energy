# torch
import torch
import torchvision

# Detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# Common Libraries
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt
from itertools import product
from datetime import date
# from google.colab.patches import cv2_imshow

# Detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.evaluation import inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data import DatasetMapper

os.chdir(os.path.join(os.getcwd(), '..', 'datasets'))

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

for d, ds in product(["train", "val"], ['fake_maxar', 'duke']):
    ds_path = os.path.join(os.getcwd(), f'{ds}_{d}', 'data')
    json_path = os.path.join(os.getcwd(), f'{ds}_{d}', 'labels.json')
    ds_name = f'{ds}_{d}'

    if ds_name in DatasetCatalog.list():
        DatasetCatalog.remove(ds_name)
        MetadataCatalog.remove(ds_name)

    register_coco_instances(ds_name, {}, json_path, ds_path)


ds_name = 'manual_maxar_val'   
ds_path = os.path.join(os.getcwd(), f'{ds_name}', 'data')
json_path = os.path.join(os.getcwd(), f'{ds_name}', 'labels.json')

if ds_name in DatasetCatalog.list():
    DatasetCatalog.remove(ds_name)
    MetadataCatalog.remove(ds_name)

register_coco_instances(ds_name, {}, json_path, ds_path)

print('Registered datasets!')

class EvalTrainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.same_data_loader = build_detection_test_loader(DatasetCatalog.get(cfg.DATASETS.TEST1),
                                                            mapper=DatasetMapper(cfg, is_train=False))
        self.same_data_eval = COCOEvaluator(cfg.DATASETS.TEST1)
        self.manual_maxar_loader = build_detection_test_loader(DatasetCatalog.get(cfg.DATASETS.TEST2),
                                                            mapper=DatasetMapper(cfg, is_train=False))
        self.manual_maxar_eval = COCOEvaluator(cfg.DATASETS.TEST2)


    def after_step(self):
        super().after_step()

        if self.iter % self.cfg.TEST.INTERVAL == 0:                                   

            results = inference_on_dataset(self.model, 
                                                     self.manual_maxar_loader,
                                                     self.manual_maxar_eval)

            with open(
                os.path.join(
                    self.cfg.OUTPUT_DIR,
                    'eval_manualsdata_'+str(self.cfg.DATASETS.TRAIN)+'_iter_'+str(self.iter)+'.json'), 
                    'w') as out:
                json.dump(results, out)   
            
            same_data_results = inference_on_dataset(self.model, 
                                                     self.same_data_loader,
                                                     self.same_data_eval)


            with open(
                os.path.join(
                    self.cfg.OUTPUT_DIR,
                    'eval_samedata_'+str(self.cfg.DATASETS.TRAIN)+'_iter_'+str(self.iter)+'.json'),
                    'w') as out:
                json.dump(same_data_results, out)                                

print('Defined trainer class!')


def do_train(train_dataset='duke'):

    cfg = get_cfg()

    frcnn= 'faster_rcnn_R_101_FPN_3x.yaml'

    # From Detectron2 Model Zoo
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/"+frcnn))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/"+frcnn)

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2   # for R-CNN Models

    cfg.defrost()
    cfg.DATASETS.TRAIN = (train_dataset+'_train')
    cfg.DATASETS.TEST1 = (train_dataset+'_val')
    cfg.DATASETS.TEST2 = ('manual_maxar_val')
    cfg.TEST.INTERVAL = 10_000
    cfg.SOLVER.MAX_ITER = 2_000_000
    cfg.SOLVER.STEPS = (500_000, 1_000_000, 1_500_000)

    model_name = "PISA_" + str(date.today()) + '_' + train_dataset
    cfg.OUTPUT_DIR = '../models/' + model_name

    cfg.freeze()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = EvalTrainer(cfg) 
    trainer.resume_or_load(resume=False)

    trainer.train()


for ds in ['duke', 'fake_maxar']:
    print('Commencing training on '+ds+'!') 
    do_train(ds)
