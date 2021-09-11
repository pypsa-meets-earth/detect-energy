import cv2
import os
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow


def filter_images(path, fn, save_file="", black_threshold=.25, cloudy_threshold=.65):
    """
    removes images that are
        - too cloudy or blurry based on singular values
        - (at least) partially black based on sum of all pixels
    ----------
    Arguments:
        path : (str)
            path to directory of interest
        fn : (str)
            name of dataframe storing examples
        save_file : (str)
            desired prefix to filename
        black_threshold : (0 < float < 1)
            (lower) bound on acceptable avg pixel darkness
        cloudy_threshold : (0 < float < 1)
            (upper) bound on ratio of sum(large) vs sum(all) singular vals
    """

    # get dataframe of images
    images_gdf = gpd.read_file(path + fn)

    total_sums = []

    images_gdf["to_delete"] = np.zeros(len(images_gdf))

    for idx, filename in images_gdf["filename"].iteritems():
        
        print("Considering Example {}".format(idx+1))
        
        # only consider greyscales for this analysis
        img = cv2.imread(path + filename, 0)
        
        print("Index {}, Filename {}".format(idx, path+filename))
        cv2_imshow(img)
        
        # filter black
        img = img / 255.
        total_sum = img.sum() / 256**2
        
        if total_sum < black_threshold:
            print("Black area detected!")
            images_gdf.at[idx, "to_delete"] = 1.

        # filter blurry and cloudy
        _, s, _ = np.linalg.svd(img)        
        sv_num = img.shape[0] // 50
        ratio = s[:sv_num].sum() / s.sum()

        if ratio > cloudy_threshold:
            print("Cloudy image detected!")
            images_gdf.at[idx, "to_delete"] = 1.
            
        # intermediate saving
        if (idx+1)%30 == 0:
            images_gdf.to_file("examples_" + save_file +".geojson", driver="GeoJSON")

    # delete useless images
    for idx, row in images_gdf.iterrows():
        if row["to_delete"] == 1.:
            os.remove(row["filename"])

    # remove redundant part of dataframe
    images_gdf = images_gdf.drop(images_gdf[images_gdf["to_delete"] == 1.].index)
    images_gdf = images_gdf.drop(columns=["to_delete"])
      
    print("Concluded Filtering. Remaining Examples: {}".format(len(images_gdf))) 
    images_gdf.to_file(fn + save_file, driver="GeoJSON")
