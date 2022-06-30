global resolutions

# precomputed resolutions of raster files by region
resolutions = {
        'arizona': 0.1521273311113449,
        'sudan': 0.2941356391155026,
        'china': 0.2789294695267606,
        'rotorua': 0.1254966590856217,
        'brazil': 0.4787769055150278,
        'hartford': 0.0761820560981825,
        'mexico': 0.1522336654793647,
        'kansas': 0.1520310125331718,
        'clyde': 0.1535755375821943,
        'wilmington': 0.1522415121091316,
        'dunedin': 0.1229411434163525,
        'gisborne': 0.1253881937589445,
        'palmertson': 0.1255661866324052,
        'tauranga': 0.1254967811385344,
        }

import shutil
import fiftyone as fo
import os
import sys
import pandas as pd
import geopandas as gpd
import numpy as np
from osgeo import gdal
from PIL import Image
from shapely.geometry import Polygon
import warnings
warnings.filterwarnings('ignore')

# basically a test if this is running in colab
if 'content' in os.getcwd():
    os.chdir('/content/drive/MyDrive/PyPSA_Africa_images')

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

duke_path = os.environ.get('PROJECT_DUKE_IMAGES')
datasets_path = os.environ.get('PROJECT_DATASETS')
root_path = os.environ.get('PROJECT_ROOT')
sys.path.append(root_path)

from src.utils.dataset_utils import fix_annots, fix_filenames
from src.utils.image_utils import downsample

assert duke_path is not None, f'Could not locate .env file. Got duke_path {duke_path}'
assert datasets_path is not None, f'Could not locate .env file. Got datasets_path {datasets_path}'

def extract_duke_dataset( 
                         target_base_dir,
                         dirs=None,
                         size=512, 
                         base_path="", 
                         train_ratio=0.8,
                         target_resolution=None,
                         bbox_threshold=None,
                         tower_types=['DT', 'TT', 'OT'],
                         ):
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
        dirs : (list of str or None)
            list with names of directories which satisfy the outlined structure
            if None a list of all countries in the duke dataset is inserted
        target_base_dir (str):
            directory where resulting datasets should be stored
        size : (int)
            width and height of resulting example-images (images are made quadratic)
        base_path : (str)
            path to directories from which all dirs are accessible
        train_ratio : (float)
            share of examples labelled as part of training set (rest is val set),
        target_resolution: (None or float)
            if float, resolutions are scaled to target_resolution
        bbox_threshold(None or float):
            if float, only towers are included that have bboxes which both for height and width are
            height 
        tower_types(List[str]):
            list of tower types that are taken into the dataset
                TT: transmission towers
                DT: distribution towers
                TT: other towers
        
    ----------
    Returns:
        -
    """

    if dirs is None:
        dirs = [ 
                #'hartford',   #  (APPEARS TO HAVE CORRUPTED GEOJSON FILES)
                'china',
                'kansas',
                'dunedin',
                'gisborne',
                'palmertson',
                'rotorua',
                'tauranga',
                'wilmington',
                'arizona',
                'clyde',
                'sudan',
                'mexico',
                'brazil',
                ]

    prefixes = [word[:2].upper() for word in dirs]

    width, height = size, size

    if bbox_threshold is None:
        bbox_threshold = 0.
    
    label = 'tower'

    out_train = 'train'
    out_val = 'val' 

    print('Starting dataset extraction')
    print('Note currently all towers are labelled as tower')

    if not os.path.isdir(target_base_dir): 
        os.mkdir(target_base_dir)

    for country, prefix in zip(dirs, prefixes):
        
        # set up working path
        print("Extracting images from {}...".format(country))

        print(f'Assuming resolution {resolutions[country]} m/pixel')
        res = resolutions[country]

        country_out = os.path.join(target_base_dir, country)
        if not os.path.isdir(country_out): 
            os.mkdir(country_out)

        train_path = os.path.join(country_out, out_train)
        val_path = os.path.join(country_out, out_val)

        if not os.path.isdir(train_path): 
            os.mkdir(train_path)
        if not os.path.isdir(val_path): 
            os.mkdir(val_path)

        # train_path = out_train

        os.chdir(os.path.join(os.getcwd(), country, "raw"))
        
        # setup directory for resulting images
        # set up resulting dataset of examples (with towers)
        tower_df = gpd.GeoDataFrame({"filename": [], 
                                "ul_x": [], "ul_y": [], "lr_x": [], "lr_y": [], 
                                })
        
        # set up datasets for current country
        try: 
            dataset_train = fo.Dataset(name=country+'_'+out_train)
        except:
            dataset_train = fo.load_dataset(country+'_'+out_train)
            dataset_train.delete()
            dataset_train = fo.Dataset(name=country+'_'+out_train)
        dataset_train.persistent = False

        try: 
            dataset_val = fo.Dataset(name=country+'_'+out_val)
        except:
            dataset_val = fo.load_dataset(country+'_'+out_val)
            dataset_val.delete()
            dataset_val = fo.Dataset(name=country+'_'+out_val)
        dataset_val.persistent = False
        
        # Starting with adding examples to the training set
        curr_path = train_path
        curr_dataset = dataset_train
        switched_already = False

        # create list of relevant files
        filelist = os.listdir()
        csv_files = [fn for fn in filelist if fn.endswith('.csv')]

        unders = [i for i, letter in enumerate(csv_files[0]) if letter is "_"]
        
        file_prefix = csv_files[0][:unders[-1]+1]
        num_files = len(csv_files)

        tif_files = [file_prefix + str(i+1) + '.tif' for i in range(num_files)]  
        geojson_files = [file_prefix + str(i+1) + '.geojson' for i in range(num_files)]  

        if target_resolution is not None:
            scaled_size = int(size * target_resolution / res)
            print(f'Pictures from {country} will have {scaled_size}x{scaled_size} pixels.')
            scaled_width, scaled_height = scaled_size, scaled_size
        else:
            scaled_height, scaled_width = height, width
        
        # iterate over files
        for i, (tif, geojson) in enumerate(zip(tif_files, geojson_files)):        

            if (i+1) / num_files > train_ratio and not switched_already: 
                print('Switching to mode val after {} of {} files due to train ratio'.format(
                      i, num_files, train_ratio, train_ratio))
        
                print(base_path, country, curr_path)
                export_dir = curr_path

                # Export training dataset
                if len(curr_dataset) > 0:
                    curr_dataset.export(
                        export_dir=export_dir,
                        dataset_type=fo.types.COCODetectionDataset,
                        label_field='ground_truth',
                        )
                else:
                    print(f'Did not export empty dataset for training in {country}')
                
                curr_path = val_path
                curr_dataset = dataset_val
                switched_already = True


            print("Opening geojson file: ", geojson)
            # open files and get bands
            try:
                annots = gpd.read_file(geojson)
            except:
                print("Unable to read annotation file {}".format(geojson))
                print("Continuing to the next file...")
                continue

            # make sure geojson contains information
            if len(annots.columns) == 1:
                print('Bad geojson detected! Continuing...') 
                continue

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

            # make sure the dataframe contains only towers
            annots = annots[annots['geometry'].apply(lambda x: isinstance(x, Polygon))]
            if annots.empty: continue

            annots["ul"], annots["lr"], annots['geometry'] = zip(*annots['pixel_coordinates'].map(to_pixels))

            tif_width, tif_height = info['size'][0], info['size'][1]

            for _, tower in annots.iterrows():

                if not isinstance(tower.geometry, Polygon): continue
                
                if not tower['label'] in tower_types:
                    continue

                example_name = prefix + '_' + str(np.random.randint(1e10, 1e11)) + '.png'

                # define the bounds of random offset
                bb_ul, bb_lr = tower['ul'], tower['lr']
                min_x, max_x = max(0, bb_lr[0] - scaled_width), min(bb_ul[0], tif_width - scaled_width)
                min_y, max_y = max(0, bb_lr[1] - scaled_height), min(bb_ul[1], tif_height - scaled_height)

                # randomly draw corner of image (this can fail if towers are close to the frame -> skip tower)
                try:
                    img_ul_x = np.random.randint(min_x, max_x)
                    img_ul_y = np.random.randint(min_y, max_y)
                except:
                    continue

                # determine bounding box relative to new image
                bb_ul -= np.array([img_ul_x, img_ul_y])
                bb_lr -= np.array([img_ul_x, img_ul_y])

                # add main tower in image 
                outer_bbox = [bb_ul[0], bb_ul[1], bb_lr[0]-bb_ul[0], bb_lr[1]-bb_ul[1]]
                outer_bbox = (np.array(outer_bbox) / scaled_width).tolist()

                if outer_bbox[2] < bbox_threshold or outer_bbox[3] < bbox_threshold:
                    continue

                # set up image and new filename
                new_img = np.zeros((scaled_height, scaled_width, 3), dtype=np.uint8)
        
                # transfer pixel data
                try:
                    for i in range(3):
                        new_img[:,:,i] = bands[i].ReadAsArray(img_ul_x, img_ul_y, scaled_width, scaled_height)
                except:
                    continue

                # transform array to image
                new_img = downsample(new_img, target_size=(size, size))

                img = Image.fromarray(new_img, 'RGB')
                img.save(os.path.join('./../', curr_path, example_name), quality=100)

                # add to dataset
                sample = fo.Sample(filepath=os.path.join(
                                   base_path, country, curr_path, example_name)
                                   )
                
                detections = []
                detections.append(fo.Detection(label=label, bounding_box=outer_bbox))

                # create Polygon of created image
                img_corner = np.array([img_ul_x, img_ul_y])
                img_polygon = Polygon([
                                    img_corner,
                                    img_corner + np.array([scaled_width, 0]),
                                    img_corner + np.array([scaled_width, scaled_height]),
                                    img_corner + np.array([0, scaled_height])
                                    ])

                # add secondary towers that happen to be in the same image
                for j, other in annots.iterrows():

                    if other['geometry'] == tower['geometry']: 
                        continue

                    if not other['label'] in tower_types:
                        continue

                    #if img_polygon.contains(other["geometry"]):
                    if img_polygon.intersects(other["geometry"]):

                        ul_pixels = np.min(other['geometry'].exterior.xy, axis=1)
                        lr_pixels = np.max(other['geometry'].exterior.xy, axis=1)

                        ul = (ul_pixels - img_corner) / scaled_width
                        lr = (lr_pixels - img_corner) / scaled_width
                        w, h = lr - ul

                        bbox = [ul[0], ul[1], w, h]

                        if not img_polygon.contains(other['geometry']):
                            in_part = other['geometry'].intersection(img_polygon)
                            shared_fraction = in_part.area / other['geometry'].area

                            bbox[0] = max(bbox[0], 0)
                            bbox[1] = max(bbox[1], 0)
                            bbox[2] = min(bbox[2], 1 - bbox[0])
                            bbox[3] = min(bbox[3], 1 - bbox[1])
                        
                        else:
                            shared_fraction = 1

                        if bbox[2] < bbox_threshold or bbox[3] < bbox_threshold:
                            continue

                        if not bbox == outer_bbox and shared_fraction > 0.5:
                            detections.append(fo.Detection(label=label, bounding_box=bbox))
                
                sample["ground_truth"] = fo.Detections(detections=detections)
                
                curr_dataset.add_sample(sample)
                
        export_dir = curr_path

        # Export training dataset
        try:
            curr_dataset.export(
                    export_dir=export_dir,
                    dataset_type=fo.types.COCODetectionDataset,
                    label_field='ground_truth',
                    )
        except ValueError:
            print(f'Could not export: {export_dir}; Length: {len(curr_dataset)}')
            print('Continuing...')
            
        fix_filenames(os.path.join(base_path, country, out_val, 'labels.json'))

        os.chdir(os.path.abspath(os.path.join('', '../..')))

    print('Done with all regions. Proceeding to packaging...')
    ds_name = target_base_dir.split('/')[-1]

    for mode in ['train', 'val']:

        datasets = []
        export_dir = os.path.join(target_base_dir, ds_name+'_'+mode)

        for country in dirs:

            ds_path = os.path.join(target_base_dir, country, mode)

            if not os.path.isdir(ds_path):
                continue

            try:
                datasets.append(fo.Dataset.from_dir(
                        dataset_type=fo.types.COCODetectionDataset,
                            data_path=os.path.join(ds_path, 'data'),
                            labels_path=os.path.join(ds_path, 'labels.json')))
            except ValueError:
                continue 

        if len(datasets) == 0:
            print(f'Could not create dataset in mode {mode} for {ds_name}')
            continue
         
        ds = datasets[0]
        for curr_ds in datasets[1:]:
            ds.merge_samples(curr_ds) 

        print('Exporting...')
        ds.export(
                    export_dir=export_dir,
                    dataset_type=fo.types.COCODetectionDataset,
                    label_field='ground_truth',
                    )
        print(f'Saved dataset to {export_dir}')

        fix_annots(os.path.join(export_dir, 'labels.json'))

    print('Done with merging dataset - only cleanup remaining')

    for country in dirs:
        
        to_delete = os.path.join(target_base_dir, country)
        if os.path.isdir(to_delete):
            shutil.rmtree(to_delete)

    print('Prepared datasets!')
    train_path = os.path.join(target_base_dir, ds_name+'_train')
    val_path = os.path.join(target_base_dir, ds_name+'_val')
    print(f'Find training set at: {train_path}')
    print(f'Find validation set at: {val_path}')


if __name__ == "__main__":

    dataset_name = 'testset'
    target_base_dir = os.path.join(datasets_path, dataset_name)
    os.chdir(duke_path)
    extract_duke_dataset(
                         target_base_dir, 
                         size=512,
                         train_ratio=0.8,
                         base_path=duke_path,
                         target_resolution=0.35,
                         # bbox_threshold=0.025,
                         tower_types=['TT', 'DT']
                        )