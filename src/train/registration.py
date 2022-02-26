import os
import sys
from itertools import product
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances 
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
DATASETS_PATH = os.environ.get('PROJECT_DATASETS')

def register_all():
    # register used datasets
    ds_names = ['fake_maxar', 'maxar']
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
    ds_name = f'{name}_{mode}'
    json_path = os.path.join(DATASETS_PATH, f'{ds_name}/labels.json')
    ds_path = os.path.join(DATASETS_PATH, f'{ds_name}/data/')

    if ds_name in DatasetCatalog.list():
        DatasetCatalog.remove(ds_name)
        MetadataCatalog.remove(ds_name)

    register_coco_instances(ds_name, {}, json_path, ds_path)