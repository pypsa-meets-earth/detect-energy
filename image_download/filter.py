
#%%
import os
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#%%
def im2np(image_path):
    # Read Image as Grayscale
    im = Image.open(image_path).convert('L')
    # Convert Image to Numpy as array 
    im = np.array(im)
    return im, image_path

def get_black_border(img,tol=0):
    # img is 2D image data
    # tol  is tolerance (between 0 and 255, 0 for completely black)
    mask = img>tol
    height,width = img.shape
    mask0,mask1 = mask.any(0),mask.any(1)
    x_start,x_end = mask0.argmax(),width-mask0[::-1].argmax()
    y_start,y_end = mask1.argmax(),height-mask1[::-1].argmax()
    # return y_start,y_end,x_start,x_end
    if (y_start, y_end) == (0,height) and (x_start,x_end) == (0, width):
        return -1 # No Black Border
    else:
        return 0 # Black Border



#%%
def filter_images(path, fn, save_file="_filtered", black_threshold=0, cloudy_threshold=.65):
    """
    adds filter column based on images that
        - have a black border
        - are too cloudy or blurry based on singular values  (2)

    ----------
    Arguments:
        path : (str)
            path to directory of interest
        fn : (str)
            name of dataframe storing examples
        save_file : (str)
            desired prefix to filename
        black_threshold : (0 < int < 255)
            (upper) bound on acceptable pixel value (0 for black)
        cloudy_threshold : (0 < float < 1)
            (upper) bound on ratio of sum(large) vs sum(all) singular vals
    """

    # get dataframe of images
    images_gdf = gpd.read_file(path + fn)

    images_gdf["filter"] = np.zeros(len(images_gdf))

    for idx, filename in images_gdf["filename"].iteritems():
        
        print("Considering Example {}".format(idx+1))
        
        # only consider greyscales for this analysis
        img, _ = im2np(path + filename)
        
        print("Index {}, Filename {}".format(idx, path+filename))

        # filter black border
        if get_black_border(img, tol = black_threshold) != -1:
            print("Black border detected!")
            images_gdf.at[idx, "filter"] = 1

        # filter blurry and cloudy
        _, s, _ = np.linalg.svd(img)        
        sv_num = img.shape[0] // 50
        ratio = s[:sv_num].sum() / s.sum()

        if ratio > cloudy_threshold:
            print("Cloudy image detected!")
            images_gdf.at[idx, "filter"] = 2
            
        # intermediate saving
        if (idx+1)%30 == 0:
            images_gdf.to_file("examples_" + save_file +".geojson", driver="GeoJSON")

    # delete useless images
    # for idx, row in images_gdf.iterrows():
    #     if row["to_delete"] == 1.:
    #         os.remove(row["filename"]+'.png')

    # remove redundant part of dataframe
    # images_gdf = images_gdf.drop(images_gdf[images_gdf["to_delete"] == 1.].index)
    # images_gdf = images_gdf.drop(columns=["to_delete"])
      
    print("Concluded Filtering. Remaining Examples: {}".format(len(images_gdf))) 
    images_gdf.to_file(fn + save_file, driver="GeoJSON")

#%%
filter_images('examples/', "tower_examples.geojson")
# %%
