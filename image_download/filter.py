import cv2
import glob
import os
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from google.colab.patches import cv2_imshow

def filter_images(path, black_threshold=.25, cloudy_threshold=.65):
    """
    removes images that are
        - too cloudy or blurry based on singular values
        - (at least) partially black based on sum of all pixels
    ----------
    Arguments:
        path : (str)
            path to directory of interest
        black_threshold : (0 < float < 1)
            (lower) bound on acceptable avg pixel darkness
        cloudy_threshold : (0 < float < 1)
            (upper) bound on ratio of sum(large) vs sum(all) singular vals
    """

    # get dataframe of images
    images_gdf = glob.glob(path + "*.geojson")[0]
    images_gdf = gpd.read_file(images_gdf)

    total_sums = []

    def delete_example(df, idx):
        """
        Deletes row in df and the respective .png file
        """
        fn = df.iloc[idx]["filename"]
        os.remove(path + fn + '.png')

        return df.drop([df.index[idx]])  


    for idx, filename in images_gdf["filename"].iteritems():
        
        print("Considering Example {}".format(idx+1))
        
        # only consider greyscales for this analysis
        img = cv2.imread(path + filename + ".png", 0)
        
        print("Showing image with index {}".format(idx))
        cv2_imshow(img)
        
        # filter black
        img = img / 255.
        total_sum = img.sum() / 256**2
        
        if total_sum < black_threshold:
            print("Black area detected!")
            images_gdf = delete_example(images_gdf, idx)
            continue

        # filter blurry and cloudy
        _, s, _ = np.linalg.svd(img)        
        sv_num = img.shape[0] // 50
        ratio = s[:sv_num].sum() / s.sum()

        if ratio > cloudy_threshold:
            print("Cloudy image detected!")
            images_gdf = delete_example(images_gdf, idx)
            continue

    print("Concluded Filtering. Remaining Examples: {}".format(len(images_gdf))) 


if __name__ == "__main__":
    filter_images("../examples/")
