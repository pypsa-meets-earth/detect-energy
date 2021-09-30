import os
import pandas as pd
import geopandas as gpd
import numpy as np
from osgeo import gdal
from itertools import product
from PIL import Image
import fiftyone as fo
from shapely.geometry import Polygon



def extract_duke_dataset(dirs, prefixes, imgs_per_tower=2, width=512, height=512, base_path=""):
    """
    Extracts training images and bounding from region-zips provided in
    'https://figshare.com/articles/dataset/Electric_Transmission_and_
    Distribution_Infrastructure_Imagery_Dataset/6931088'
    
    Iterates over a list of directories and creates examples if it encounters the
    following structure in these directories:
        [dir_name]/raw/[.tif, .csv, .geojson etc. files]
        stores images to the following structure
        [dir_name]/examples/[prefix]+[id]+".png"]
        and a summarizing geojson file storing a dataframe of filenames and bbox
        [dir_name]/examples/[prefix]+"examples.geojson"
    
    Data must be unzipped!
    Be sure that all directories in [dirs] are in os.getcwd()!
    
    ----------
    Arguments:
        dirs : (list of str)
            list with names of directories which satisfy the outlined structure
        prefixes : (list of str)
            list with respective prefixes of respective resulting .png and .geojson files
        imgs_per_tower : (int)
            number of examples created for every tower found in geojson files
        width : (int)
            width of resulting example-images
        height : (int)
            height of resulting example-images
        base_path : (str)
            path to directories from which all dirs are accessible
    ----------
    Returns:
        -
    """

    for country, prefix in zip(dirs, prefixes):
        
        # set up working path
        print("Extracting images from {}...".format(country))
        os.chdir(os.path.join(os.getcwd(), country, "raw"))
        
        # setup directory for resulting images
        example_path = "./../examples/"
        if not os.path.isdir(example_path): os.mkdir(example_path)

        # set up resulting dataset of examples (with towers)
        tower_df = gpd.GeoDataFrame({"filename": [], 
                                "ul_x": [], "ul_y": [], "lr_x": [], "lr_y": [], 
                                #"geometry": []
                                })
        
        # set up dataset for current country
        try: 
           dataset = fo.Dataset(name=country)
        except:
            dataset = fo.load_dataset(country)
            dataset.delete()
            dataset = fo.Dataset(name=country)
            # _dataset2 = fo.load_dataset("my_second_dataset")
        dataset.persistent = False

        # create list of relevant files
        filelist = os.listdir()
        csv_files = [fn for fn in filelist if fn.endswith('.csv')]

        unders = [i for i, letter in enumerate(csv_files[0]) if letter is "_"]
        file_prefix = csv_files[0][:unders[-1]+1]
        num_files = len(csv_files)

        csv_files = [file_prefix + str(i+1) + '.csv' for i in range(num_files)]  
        tif_files = [file_prefix + str(i+1) + '.tif' for i in range(num_files)]  
        geojson_files = [file_prefix + str(i+1) + '.geojson' for i in range(num_files)]  

        # iterate over files
        for csv, tif, geojson in zip(csv_files, tif_files, geojson_files):        

            print("Opening geojson file: ", geojson)
            # open files and get bands
            annots = gpd.read_file(geojson)
            assets = gpd.read_file(csv)
            ds = gdal.Open(tif)
            bands = [ds.GetRasterBand(i) for i in range(1, 4)]
            info = gdal.Info(tif, format="json")

            pd.set_option('display.max_columns', None)

            # remove all assets except towers            
            remove_assets = ["DL", "TL", "OL", "SS"]
            for to_remove in remove_assets:
                annots = annots[annots["label"] != to_remove]

            def to_pixels(geom):
                '''
                receives pixel coordinates as string and returns columns 
                upper left, lower right and geometry as Polygon (rectangular) 
                all coordinates are relative to the tif file the assets is in
                '''
                geom = geom.split(" ")
                geom.remove('[')
                geom.remove(']')

                # transform to Polygon with rectangular bbox
                geom = [entry for entry in geom if not '[' in entry and not ']' in entry]
                geom = [int(float(entry.replace(",", ""))) for entry in geom]
                x, y = geom[::2], geom[1::2] 
                geom = Polygon([[max(x), max(y)], [max(x), min(y)], [min(x), min(y)], [min(x), max(y)]])
                return np.array([min(x), min(y)]), np.array([max(x), max(y)]), geom

            annots["ul"], annots["lr"], annots['geometry'] = zip(*annots['pixel_coordinates'].map(to_pixels))

            tif_width, tif_height = info['size'][0], info['size'][1]

            for (curr, tower), i in product(annots.iterrows(), range(imgs_per_tower)):

                example_name = prefix + '_' + str(np.random.randint(1e10, 1e11)) + '.png'

                # define the bounds of random offset
                bb_ul, bb_lr = tower['ul'], tower['lr']
                min_x, max_x = max(0, bb_lr[0] - width), min(bb_ul[0], tif_width - width)
                min_y, max_y = max(0, bb_lr[1] - height), min(bb_ul[1], tif_height - height)

                # randomly draw corner of image (this can fail if towers are close to the frame -> skip tower)
                try:
                    img_ul_x = np.random.randint(min_x, max_x)
                    img_ul_y = np.random.randint(min_y, max_y)
                except:
                    continue

                # determine bounding box relative to new image
                bb_ul -= np.array([img_ul_x, img_ul_y])
                bb_lr -= np.array([img_ul_x, img_ul_y])

                # set up image and new filename
                new_img = np.zeros((height, width, 3), dtype=np.uint8)
        
                # transfer pixel data
                try:
                    for i in range(3):
                        new_img[:,:,i] = bands[i].ReadAsArray(img_ul_x, img_ul_y, width, height)
                except:
                    continue

                # transform array to image
                img = Image.fromarray(new_img, 'RGB')
                img.save(example_path + example_name, quality=100)

                # add to dataset
                sample = fo.Sample(filepath=os.path.join(
                                   base_path, country) + "/examples/" + example_name)
                detections = []

                # add main tower in image 
                bbox = [bb_ul[0], bb_ul[1], bb_lr[0]-bb_ul[0], bb_lr[1]-bb_ul[1]]
                bbox = (np.array(bbox) / width).tolist()
                detections.append(fo.Detection(label='tower', bounding_box=bbox))
                

                # create Polygon of created image
                img_corner = np.array([img_ul_x, img_ul_y])
                img_polygon = Polygon([
                                    img_corner,
                                    img_corner + np.array([width, 0]),
                                    img_corner + np.array([width, height]),
                                    img_corner + np.array([0, height])
                                    ])

                # add secondary towers that happen to be in the same image
                for j, other in annots.iterrows():
                    if img_polygon.contains(other["geometry"]):
                        ul = (other['ul'] - img_corner) / width
                        lr = (other['lr'] - img_corner) / width
                        w, h = lr - ul

                        bbox = [ul[0], ul[1], w, h]
                        detections.append(fo.Detection(label='tower', bounding_box=bbox))
                
                sample["ground_truth"] = fo.Detections(detections=detections)
                dataset.add_sample(sample)
                
        
            
         
        export_dir = os.path.join(base_path, country) + "/examples/"
        label_field = "ground_truth"  

        # Export dataset
        dataset.export(
              export_dir=export_dir,
              dataset_type=fo.types.COCODetectionDataset,
              label_field=label_field,
              )

        os.chdir(os.path.abspath(os.path.join('', '../..')))

        break



if __name__ == "__main__":
    base_path = "/content/drive/MyDrive/PyPSA_Africa_images/"
    os.chdir(base_path)
    dirs = ["china", "mexico", 'brazil']
    prefixes = ["CH", "ME", "BR"]
    extract_duke_dataset(dirs, prefixes, imgs_per_tower=1, base_path=base_path)
