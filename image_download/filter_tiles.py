import os
import geopandas as gpd
import numpy as np
from PIL import Image


def im2np(image_path):
    # Read Image as Grayscale
    im = Image.open(image_path).convert('L')
    # Convert Image to Numpy as array 
    im = np.array(im)
    return im, image_path

def get_black_border(img,black_point=0):
    # img is 2D image data
    # black_point  is pixel value for black (between 0 and 255, 0 for completely black)
    mask = img>black_point
    height,width = img.shape
    mask0,mask1 = mask.any(0),mask.any(1)
    x_start,x_end = mask0.argmax(),width-mask0[::-1].argmax()
    y_start,y_end = mask1.argmax(),height-mask1[::-1].argmax()
    return y_start,y_end,x_start,x_end

def isBlackBorder(img,black_point=0):
    height,width = img.shape
    y_start,y_end,x_start,x_end = get_black_border(img,black_point)
    if (y_start, y_end) == (0,height) and (x_start,x_end) == (0, width):
        return False
    else:
        return True

# Simple Cloud Cover function
def isCloudy(im, white_point=101, thresh = 0.65):
    #TODO: Dynamically set white_point
    height,width = im.shape
    # Put threshold to make it binary
    binarr = np.where(im>white_point, 1, 0)
    # Find Total sum of 2D array thresh
    total = sum(map(sum, binarr))
    ratio = total/height/width
    if ratio > thresh:
        return True
    else:
        return False 

def isBlurry(img, thresh=.65):
    _, s, _ = np.linalg.svd(img)        
    sv_num = img.shape[0] // 50
    ratio = s[:sv_num].sum() / s.sum()
    if ratio > thresh:
        return True
    else:
        return False

def remove_images(filtered_df, img_dir):
        f_df = filtered_df
        # Delete Image File
        for idx, row in f_df.iterrows():
            filter_value = row["filter"]
            fn = row["filename"]
            if filter_value > 0:
                print(f"removing {fn}, filter = {filter_value}")
                fn = row["filename"]
                file_path = os.path.join(img_dir,row["filename"])
                # print(os.path.exists(file_path))
                # os.remove(file_path)
                # Move files instead of deleting for dev 
                delete_dir = os.path.join(img_dir,"filter_delete")
                os.makedirs(delete_dir,exist_ok=True)
                os.rename(file_path, os.path.join(delete_dir,fn)) # Move file to delete_dir
        # remove redundant part of dataframe
        f_df = f_df.drop(f_df[f_df["filter"] > 0].index)
        f_df = f_df.drop(columns=["filter"])

        return f_df

def filter_images(gdf_path, delete_filtered=False, black_point=0, white_point=180, cloudy_threshold=.45, blurry_threshold=.65):
    """
    adds filter column based on images that
        - have a black border                       (filter = 1)
        - are too cloudy or                         (filter = 2)
        - are too blurry based on singular values   (filter = 3)

    ----------
    Arguments:
        path : (str)
            path to geojson storing examples
        delete_filtered : (bool)
            deletes filtered images from disk and saves dataframe with suffix "_clean" (True)
        black_point : (0 < int < 255)
            (upper) bound on pixel brightness (0 for black)
        white_point : (0 < int < 255)
            (lower) bound on pixel brightness (255 for white)
        cloudy_threshold : (0 < float < 1)
            (upper) bound on ratio of sum(large) vs sum(all) singular vals
        blurry_threshold : (0 < float < 1)
            (upper) bound on ratio of sum(large) vs sum(all) singular vals
    """
    suffix = "_filtered"
    img_dir = os.path.join(os.path.dirname(gdf_path),'')
    fn = os.path.basename(gdf_path)

    # get dataframe of images
    images_gdf = gpd.read_file(gdf_path)

    images_gdf["filter"] = np.zeros(len(images_gdf))

    for idx, filename in images_gdf["filename"].iteritems():
        
        # print("Considering Example {}".format(idx+1))
        
        # only consider greyscales for this analysis
        img, img_path = im2np(img_dir + filename)
        
        # filter black border
        if isBlackBorder(img,black_point) is True:
            print(f"Black Border Detected! for {img_path}")
            images_gdf.at[idx, "filter"] = 1
            continue

        # filter cloudy
        if isCloudy(img, white_point, cloudy_threshold):
            print(f"Cloudy Image Detected! for {img_path}")
            images_gdf.at[idx, "filter"] = 2
            continue

        # filter blurry
        if isBlurry(img, blurry_threshold):
            print(f"Blurry Image Detected! for {img_path}")
            images_gdf.at[idx, "filter"] = 3

            
        if (idx+1)%30 == 0:
            print("Index {}, Filename {}".format(idx, filename))
            # images_gdf.to_file("examples_" + save_file +".geojson", driver="GeoJSON")
    
    if delete_filtered is True:
        print("Deleting Filtered Images")
        suffix = "_clean"
        images_gdf = remove_images(images_gdf, img_dir)

    print("Concluded Filtering. Remaining Examples: {}".format(len(images_gdf))) 
    images_gdf.to_file(os.path.join(img_dir, fn.split('.')[0] + suffix + '.geojson'), driver="GeoJSON")

    return images_gdf

def verify_df_img(gdf_path):
    verify = True
    vdf = gpd.read_file(gdf_path)
    img_dir = os.path.dirname(gdf_path)
    # ------- Length Test
    df_list = vdf['filename'].to_list()
    img_list = [img for img in os.listdir(img_dir) if img.endswith(".png")]
    if len(img_list) != len(df_list):
        print("Length Test Failed")
        verify = False
        print(f"Number of images in {img_dir}: {len(img_list)}")
        print(f"Number of filenames in {gdf_path}: {len(df_list)}")
        
    # ------- Duplicates Test
    df_dup_len = vdf.duplicated(subset=['filename']).sum()
    if df_dup_len != 0:
        print("Duplicates Test Failed")
        verify = False
        print(f"Number of duplicates in {gdf_path}: {df_dup_len}")

    # ------- Image Existance Test
    df_set = set(df_list)
    img_set = set(img_list)
    if df_set != img_set:
        print("Image Existance Test Failed")
        verify = False
        NotInDF = img_set - df_set
        NotInDir = df_set - img_set
        print(f"Images Not in {gdf_path} : {NotInDF}")
        print(f"Images Not in {img_dir} : {NotInDir}")
    
    if verify is True:
        print("All Test Passed")

    return verify


if __name__ == "__main__":
    filter_images('examples/tower_examples.geojson', delete_filtered=True)
    verify_df_img("examples/tower_examples_clean.geojson")
    
