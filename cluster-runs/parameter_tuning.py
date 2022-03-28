import os
# os.chdir('/disk3/fioriti/git/detect_energy')
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"

import sys
# sys.path.append(os.environ.get('PROJECT_ROOT'))
data_path = os.environ.get('PROJECT_DATASETS')
model_out_path = "/disk3/fioriti/git/detect_energy/cluster-runs/models/PISA_Parameter_Tuning"


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
from detectron2.data import build_detection_test_loader, DatasetMapper


DATASETS_PATH = os.environ.get('PROJECT_DATASETS')

print("DATASETS_PATH: " + DATASETS_PATH)

def register_all():
    # register used datasets
    ds_names = ['fake_maxar', 'duke', 'duke_512']
    modes = ['train', 'val']

    for name, mode in product(ds_names, modes):

        ds_name = f'{name}_{mode}'
        json_path = os.path.join(DATASETS_PATH, f'{ds_name}/labels.json')
        ds_path = os.path.join(DATASETS_PATH, f'{ds_name}/data/')

        if ds_name in DatasetCatalog.list():
            DatasetCatalog.remove(ds_name)
            MetadataCatalog.remove(ds_name)

        register_coco_instances(ds_name, {}, json_path, ds_path)

    ds_name = 'manual_maxar_val'
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

    cfg.DATASETS.TRAIN = 'duke_512_train'
    cfg.DATASETS.TEST = ['manual_maxar_val']
    cfg.DATASETS.EVAL = ['manual_maxar_val', 'duke_512_val', 'duke_512_train']

    cfg.TEST.INTERVAL = 5_000
    cfg.SOLVER.MAX_ITER = 50_000
    cfg.SOLVER.STEPS = (40_000, 45_000)

    # setup current parameters
    cfg.SOLVER.IMS_PER_BATCH = params['SOLVER.IMS_PER_BATCH']
    cfg.SOLVER.BASE_LR = params['SOLVER.BASE_LR']
    cfg.SOLVER.MOMENTUM = params['SOLVER.MOMENTUM']
    cfg.SOLVER.WEIGHT_DECAY = params['SOLVER.WEIGHT_DECAY']
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = params['MODEL.ANCHOR_GENERATOR.SIZES']

    select_model = params.model_type.split(".")[0]

    model_name = f"MODEL_{select_model}_"
    # model_name +=f"LR_{cfg.SOLVER.BASE_LR}_"
    # model_name +=f"IMSPERBATCH_{cfg.SOLVER.IMS_PER_BATCH}_"
    # model_name +=f"MOM_{cfg.SOLVER.MOMENTUM}_"
    # model_name +=f"WEIGHTDECAY_{cfg.SOLVER.WEIGHT_DECAY}"
    model_name +=f"ANCHORS"+ \
                str(cfg.MODEL.ANCHOR_GENERATOR.SIZES).replace('[[', '_').replace(']]', '_').replace(',', '_').replace(' ', '')

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
        'SOLVER.IMS_PER_BATCH': [16],       # default
        'SOLVER.WEIGHT_DECAY': [0.0001],    # default
        'MODEL.ANCHOR_GENERATOR.SIZES': [[16, 32, 64, 128, 256]],
        }

    parameter_sweep = list(ParameterGrid(parameters))

    for params in parameter_sweep:
        run_parameters(params)