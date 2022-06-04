import json

def fix_annots(file):
    '''
    adds information on the 'iscrowd' property 
    for training with detectron2 to an annotation json file
    made by fiftyone
    '''
    dictionary = json.load(open(file)) 
    for annot in dictionary['annotations']:
        annot['iscrowd'] = 0
    
    with open(file, "w") as f:
        json.dump(dictionary, f)


def fix_filenames(file):
    '''
    removes buggy -2 attached by fiftyone to filenames
    '''

    dictionary = json.load(open(file))
    
    for imgs in dictionary['images']:
        imgs['file_name'] = imgs['file_name'].replace('-2', '')
    
    with open(file, "w") as f:
        json.dump(dictionary, f)
