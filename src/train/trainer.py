import os
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

sys.path.append(os.environ.get('PROJECT_ROOT'))

from src.train.transfer_mapper import TransferDatasetMapper


from detectron2.engine import DefaultTrainer
from detectron2.data import build_detection_test_loader, DatasetCatalog
from detectron2.data import build_detection_train_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.mapper import DatasetMapper



class DetectEnergyTrainer(DefaultTrainer):
    '''
    Trainer with (optional) capability to perform transfer learning
    ''' 

    def __init__(self, cfg):
        super().__init__(cfg)    


        # setup evaluation
        if isinstance(cfg.DATASETS.EVAL, str):
            self.eval_datasets = [cfg.DATASETS.EVAL]
        else:
            self.eval_datasets = cfg.DATASETS.EVAL 

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
        return build_detection_train_loader(cfg, 
                            mapper=TransferDatasetMapper(cfg, is_train=True))

   
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
    