from PIL import Image
from osgeo import osr, gdal
import numpy as np
from gdalconst import *
from shapely.geometry import Polygon
import geopandas as gpd
import folium
import os
import rtree
import matplotlib.pyplot as plt
# import pygeos

# os.chdir(os.path.dirname(os.path.abspath(__file__)))  # move up to parent directory

#%%
def make_polygon_list(path, resume_file=None):
    """
    Creates GeoDataFrame, checks the tif files in a desired dir 
    and adds their geometries (with respective file name) as row
    to GeoDataFrame
    -----------
    Arguments:
    path : (str)
        directory with tif files
    resume_file : (str or None)
        if None: initializes net GeoDataFrame
        otherwise loads (and resumes from) existing one
    """

    tif_files = []

    # iterate over all .tif files in desired dir
    for filename in os.listdir(path):
        if filename.endswith(".tif"): 
            tif_files.append(os.path.join(path, filename))

    # add data to existing dataframe if desired
    try:
        coverage = gpd.read_file(resume_file)
        print(f"Adding coverage to {resume_file}")
    except:
        coverage = gpd.GeoDataFrame({"filename": [], "geometry": []})

    # extract geometries of .tif files and add to geodataframe
    for file in tif_files:
        try:
            info = gdal.Info(file, format="json")
        except SystemError:
            print(f'Skipping raster {file}')
            continue
        corners = info["cornerCoordinates"]
        poly = Polygon((
                    corners["upperLeft"],
                    corners["upperRight"],
                    corners["lowerRight"],
                    corners["lowerLeft"]
                        ))
        coverage = coverage.append({"filename": file, "geometry": poly}, ignore_index=True)

    coverage = coverage.set_crs(epsg=4326)

    return coverage

#%%
def make_examples(assets,
                  coverage,
                  img_path="/../examples/",
                  max_length=10,
                  height=512,
                  width=512,
                  bbox_height=25,
                  bbox_width=30,
                  random_offset = True,
                  seed = None,
                  examples_per_tower=1):
    """
    Expects dataframe of energy infrastructure assets in fn. Iterates over df
    and for each asset finds the respective polygon from coverage geodataframe.
    Then opens the .tif file that corresponds to the polygon and cuts out the
    respective pixels. The pixels are turned into a png files and saves.
    Additionally, the respective file name, geometry, bbox, is written into
    a new GeoDataFrame
    ----------
    Arguments:
    assets : (str or GeoDataFrame)
        path to or GeoDataFrame itself of energy assets
    coverage : (str or GeoDataFrame)
        path to or GeoDataFrame itself of geometries satellite imagery
    img_path : (str)
        path to directory where resulting examples will vbe stored
    max_length : (int / None)
        restricts number of images created if desired
    height : (int)
        number of pixels in y direction
    width : (int)
        number of pixels in x direction
    random_offset : (bool)
        set false for zero offset otherwise randomly set
    seed : (int)
        seed value for numpy random generator
    examples_per_tower : (int)
        number of examples generated per tower
    ----------
    Returns:
    dataset : (GeoDataFrame)
        df of created examples. contains for each image:
            filename (ending in .png)
            for bbox: upperleft and lowerright
            geometry as Polygon
    Also
    Saves created examples as .png files to img_path
    In the same directory the GeoJSON of the respective
    GeoDataFrame dataset is stored in the same directory
    """

    # img_path = os.path.abspath("") + img_path

    if isinstance(assets, str): assets = gpd.read_file(assets)
    if isinstance(coverage, str): coverage = gpd.read_file(coverage)

    # set up resulting dataset of examples (with towers)
    dataset = gpd.GeoDataFrame({"filename": [],
                                "ul_x": [], "ul_y": [], "lr_x": [], "lr_y": [],
                                "geometry": []}).set_crs(epsg=4326)

    # bounding_box = gpd.GeoDataFrame({"filename": [], "geometry": []})

    assets = gpd.sjoin(assets, coverage, how="inner").set_crs(epsg=4326)
    assets = assets.drop(["index_right"], axis=1)

    # for image labeling
    pos = [i for i, sign in enumerate(img_path) if sign is '/'][-1]
    prefix = img_path[pos+1:]

    print(f"Maxiumum Number of Examples: {len(assets)}")
    if len(assets)<max_length:
        max_length = len(assets)
    
    if seed is not None:
        np.random.seed(seed)

    # iterate over .tif files
    for i, image in enumerate(coverage["filename"]):

        # open files
        ds = gdal.Open(image)
        info = gdal.Info(image, format="json")
        bands = [ds.GetRasterBand(i) for i in range(1, 4)]

        # extract relevant geographical data
        transform = info["geoTransform"]
        upper_left = np.array([transform[0], transform[3]])
        pixel_size = np.array([transform[1], transform[5]])

        # iterate over assets in that image
        for row in assets[assets["filename"] == image].itertuples():

            # (default=1) generates multiple examples for every examples
            for j in range(examples_per_tower):

                # compute relevant pixels
                p = row.geometry
                coords = np.array([p.xy[0][0], p.xy[1][0]])
                pixels = np.around((coords - upper_left) / pixel_size)
                pixels -= np.array([width // 2, height // 2])
                # add random offset (8 is minimal distance to image boundary)

                offset = np.zeros(2)
                if random_offset is True:
                    offset[0] = np.random.randint(-width//2 + 20, width//2 - 20)
                    offset[1] = np.random.randint(-height//2 + 20, height//2 - 20)

                pixels -= offset
                x, y = int(pixels[0]), int(pixels[1])

                # set up image and new filename
                filename = str(int(10*row.id+j))
                new_img = np.zeros((height, width, 3), dtype=np.uint8)

                # transfer pixel data
                try:
                    for i in range(3):
                        new_img[:,:,i] = bands[i].ReadAsArray(x, y, width, height)
                except:
                    continue

                # create Polygon of created image
                img_corner = upper_left + pixels*pixel_size
                img_polygon = Polygon([
                                       img_corner,
                                       img_corner + pixel_size*np.array([width,0]),
                                       img_corner + pixel_size*np.array([width,height]),
                                       img_corner + pixel_size*np.array([0,height])
                                      ])

                # get pixels of bbox; format: (ul_x, ul_y, lr_x, lr_y)
                ul = np.array([width//2, height//2]) + offset
                bbox = [
                        ul[0] - bbox_width // 2,
                        ul[1] - bbox_height // 2,
                        ul[0] + bbox_width // 2,
                        ul[1] + bbox_height // 2
                ]

                # _, ax = plt.subplots(1, 1, figsize=(8, 8))
                # ax.imshow(new_img)
                # ax.scatter([bbox[0], bbox[2]], [bbox[1], bbox[3]], s=100, c='r')
                # plt.show()

                # do not include images that are contain blank spots
                black_threshold = 10
                binarr = np.where(new_img < black_threshold, 1, 0) #black_point is upper limit
                # Find Total sum of 2D array thresh
                total = sum(map(sum, binarr))
                ratio = total/height/width

                if (ratio > 0.5).sum() == 3:
                    _, ax = plt.subplots(1, 1, figsize=(8, 8))
                    print('filtered by black borders!')
                    continue

                # do not include images that contain clouds
                white_point = 180
                # Put threshold to make it binary
                binarr = np.where(new_img>white_point, 1, 0) #white point is lower limit
                # Find Total sum of 2D array thresh
                total = sum(map(sum, binarr))
                ratio = total/height/width
                if (ratio > 0.45).sum() == 3:
                    print('filtered by cloudy!')
                    continue

                # exclude images that are blurry 
                blurr_threshold = 0.65
                # _, s, _ = np.linalg.svd(new_img)        
                _, s, _ = np.linalg.svd(new_img.sum(axis=2))        
                sv_num = new_img.shape[0] // 50
                ratio = s[:sv_num].sum() / s.sum()
                if ratio > blurr_threshold and not 'australia' in image:
                    print('filtered by blurry!')
                    continue

                # transform array to image
                img = Image.fromarray(new_img, 'RGB')
                img.save(img_path + filename + ".png", quality=100)

                # geo_bbox = gpd.GeoSeries(p).set_crs("EPSG:4326").to_crs("EPSG:3857").buffer(30)
                # geo_bbox = geo_bbox.to_crs("EPSG:4326")
                # geo_bbox = gpd.GeoSeries(p)
                # bounding_box = bounding_box.append({"filename": filename,"geometry": geo_bbox[0]},ignore_index=True)
                # bounding_box.set_crs("EPSG:4326", inplace=True)

                # add resulting image to dataset
                dataset = dataset.append({"filename": prefix + filename + '.png',
                                          "ul_x": bbox[0],
                                          "ul_y": bbox[1],
                                          "lr_x": bbox[2],
                                          "lr_y": bbox[3],
                                          "geometry": img_polygon}, 
                                          ignore_index=True)

                # # save results inbetween
                # if len(dataset)%50 == 0:
                #     dataset = dataset.set_crs(epsg=4326)
                #     dataset.to_file(os.path.join(img_path, "tower_examples.geojson"), driver="GeoJSON")
                #     bounding_box.to_file(img_path + "tower_bbox.geojson", driver="GeoJSON")
                if len(dataset)%10 == 0:
                    print("Created {} Examples!".format(len(dataset)))


                # early stoppage
                if len(dataset) == max_length:
                    prev_len = len(dataset)
                    dataset.drop_duplicates(subset=['filename'], inplace=True) # For assets that exitst in overlapping coverage areas
                    new_len = len(dataset)
                    if new_len < prev_len:
                        print(f"removed {prev_len - new_len} duplicates")
                        print(f"remaining {new_len} examples")
                    dataset.to_file(img_path + "tower_examples.geojson", driver="GeoJSON")
                    # bounding_box.to_file(img_path + "tower_bbox.geojson", driver="GeoJSON")
                    return None
                
    dataset.to_file(img_path + "tower_examples.geojson", driver="GeoJSON")   
