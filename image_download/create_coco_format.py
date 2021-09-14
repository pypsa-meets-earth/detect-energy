#%%
import os
import json
import geopandas as gpd

#%%
img_dir = "./examples/" # TODO: split dataset into /examples/train and /examples/val
examples = gpd.read_file(os.path.join(img_dir, "tower_examples.geojson"))
img_list = [img for img in os.listdir(img_dir) if img.endswith(".png")]
images = []
annotations = []
for idx, exp in examples.iterrows():

    # --- image ---
    image={}
    img_filename = str(exp.filename)
    img_path = os.path.join(img_dir, img_filename) # Not needed
    
    if img_filename not in img_list: # Checks if the png file exists
        print(f"ERROR: {img_filename} NOT FOUND in {img_dir}")

    image_id = int(exp.filename.split('.')[0])
    image["id"] = image_id
    image["width"] = 256 # TODO : Do not hardcode
    image["height"] = 256
    image["file_name"] = f"{image_id}.png"
    image["license"] = 1
    images.append(image)

    # --- annotation ---
    annotation = {}

    bbox_width = 30 # tower_width from make_examples
    bbox_height = 25 # tower_height from make_examples
    # TODO: Return width and height instead of "lr_x" and "lr_y" in make_examples

    x_top_left = exp.ul_x
    y_top_left = exp.ul_y


    annotation["id"] = image_id # Just needs a unique id, we only have one annotation per image
    annotation["image_id"] = image_id # Should be same as image["id"]
    annotation["segmentation"] = [] # Stays like this as we are not doing segmentation
    annotation["area"] = bbox_width * bbox_height # Easy
    annotation["iscrowd"] = 0 # 0 as we only have one class

    annotation["bbox"] = [x_top_left, y_top_left, bbox_width, bbox_height] # format for coco
    annotation["category_id"] = 1 # we only have one category so hardcoded

    annotations.append(annotation)

# print(image)
# print(annotation)

#%%
info={# Purely Cosmetic
    "year":2021,
    "version": "1.0",
    "description": "Satellite Imagery Dataset",
    "contributor": "PyPSA-meets-Africa",
    "url": "example.com",
    "date_created": "2021/8/28"
}

license = {# Purely Cosmetic
    "id": "1",
    "name": "Maxar?", 
    "url": "maxar.org"
}


category = {
    "id": 1, # id starts from 1
    "name": "tower",
    "supercategory": "power" # Not sure if supercategory is required
}



#%%
coco_format = {
    "info": info,
    "licenses": [license],
    "categories": [category],
    "images": images,
    "annotations": annotations
}

# print(coco_format)


# %%
with open(os.path.join(img_dir,'tower_coco.json'), 'w') as fp:
    json.dump(coco_format, fp)