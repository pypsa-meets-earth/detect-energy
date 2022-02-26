from google.colab.patches import cv2_imshow

from detectron2.utils.visualizer import Visualizer
from detectron2.structures import BoxMode
from detectron2.data import detection_utils as utils
from detectron2.data import MetadataCatalog

import torch
import matplotlib.pyplot as plt
from scipy.special import softmax
from scipy.special import expit
plt.style.use('bmh')


def change_classes(data):
    for i, datum in enumerate(data):
        # print(datum['instances'])
        classes = getattr(datum['instances'], 'gt_classes')
        setattr(datum['instances'], 'gt_classes', torch.ones_like(classes, dtype=int))
        data[i] = datum
    
    return data

def show_data(datalist, metadata=MetadataCatalog.get('fake_maxar_val'), final_output=True):

    if final_output:
        try:
            datalist = change_classes(datalist)
        except KeyError:
            pass
    
    for data in datalist:
        img = data["image"].permute(1, 2, 0).cpu().detach().numpy()
        img = utils.convert_image_to_rgb(img, cfg.INPUT.FORMAT)        

        visualizer = Visualizer(img, metadata=metadata, scale=0.6)

        if 'instances' in data:
            target_fields = data["instances"].get_fields()

            if 'proposal_boxes' in target_fields:

                objectness = [expit(val.cpu()) for val in target_fields['objectness_logits']]

                for prob, (i, box) in zip(objectness, enumerate(target_fields['proposal_boxes'].to('cpu'))):

                    if prob < max(objectness):
                        continue

                    visualizer.draw_box(box)
                    visualizer.draw_text(prob.item(), tuple(box[:2].numpy()))

                cv2_imshow(visualizer.get_output().get_image()[:,:,::-1])
                return

            elif 'gt_classes' in target_fields:
                labels = [metadata.thing_classes[i] for i in target_fields["gt_classes"]]
                visualizer = visualizer.overlay_instances(
                    labels=labels,
                    boxes=target_fields.get("gt_boxes", None),
                    masks=target_fields.get("gt_masks", None),
                    keypoints=target_fields.get("gt_keypoints", None),
                )
            else:
                visualizer = visualizer.draw_instance_predictions(data['instances'])

            cv2_imshow(visualizer.get_image()[:,:,::-1])
        
        else:
            cv2_imshow(visualizer.get_output().get_image()[:,:,::-1])


def filter_instances(data):
    '''
    Removes all instances in data that are faulty.
    E.g. have no extent

    Args:
        data(List[Dict]): each list entry is one image
    '''

    for i, datum in enumerate(data):
        
        try:
            insts = datum['instances']
        except KeyError:
            continue

        # filter instances that have area zero
        t = insts.gt_boxes.tensor.cpu()
        num = len(insts)
        mask = (t[:,0] - t[:,2]) != torch.zeros(num)
        mask *= (t[:,1] - t[:,3]) != torch.zeros(num)
        insts = insts[mask]

        datum['instances'] = insts

    return data