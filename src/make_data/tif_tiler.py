#%%
import os
import rtree
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import PIL.ImageDraw as ImageDraw

from osgeo import gdal
from gdalconst import *
from shapely.geometry import Polygon

from projections import lon_lat_to_pixel, pixel_to_lon_lat
from make_examples import make_polygon_list
import geopandas as gpd

# import pygeos

# os.chdir(os.path.dirname(os.path.abspath(__file__)))  # move up to parent directory

#%%
def lonlat2tile(raster_dataset, lon_lat, width, height):
    tile_size = np.array([width, height])
    point_px = np.array(lon_lat_to_pixel(raster_dataset,lon_lat)) # Convert lon lat coordinates to pixel coordinates on raster
    tile_col_row = point_px//tile_size # colum and row number that contains point
    tile_px = tile_col_row*tile_size # top left corner in pixel coordinates
    return tile_px


#%%
def get_tile(raster_dataset, px_col_row, width, height):
    ds = raster_dataset

    # read raster bands
    bands = [ds.GetRasterBand(i) for i in range(1, 4)]

    # set up image and new filename
    new_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Top left corner of tile
    x, y = px_col_row[0].item(), px_col_row[1].item()

    # transfer pixel data
    for i in range(3):
        new_img[:,:,i] = bands[i].ReadAsArray(x, y, width, height)

    img = Image.fromarray(new_img, 'RGB')
    return img



#%%
def make_tiled_examples(assets,
                  coverage,
                  img_path="/../examples/",
                  max_length=10,
                  height=256,
                  width=256,
                  draw_bb=False):
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
        path to directory where resulting examples will be stored
    max_length : (int / None)
        restricts number of images created if desired
    height : (int)
        number of pixels in y direction
    width : (int)
        number of pixels in x direction
    random_offset : (bool)
        set false for zero offset otherwise randomly set
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

    img_path = os.path.abspath("") + img_path

    if isinstance(assets, str): assets = gpd.read_file(assets)
    if isinstance(coverage, str): coverage = gpd.read_file(coverage)

    # set up resulting dataset of examples (with towers)
    dataset = gpd.GeoDataFrame({"filename": [],
                                "ul_x": [], "ul_y": [], "lr_x": [], "lr_y": [],
                                "geometry": []})

    # bounding_box = gpd.GeoDataFrame({"filename": [], "geometry": []})

    assets = gpd.sjoin(assets, coverage, how="inner")
    assets = assets.drop(["index_right"], axis=1)

    # some hacky code to set filename
    pos = [i for i, sign in enumerate(img_path) if sign is '/'][-1]
    prefix = img_path[pos+1:] + '_'

    print(f"Maxiumum Number of Examples: {len(assets)}")
    if len(assets)<max_length:
        max_length = len(assets)

    # iterate over .tif files
    for i, sat_img in enumerate(coverage["filename"]):

        # open files
        ds = gdal.Open(sat_img)

        # iterate over assets in that image
        for row in assets[assets["filename"] == sat_img].itertuples():

                # compute relevant pixels
                asset_geom = row.geometry
                asset_point = np.array([asset_geom.xy[0][0], asset_geom.xy[1][0]])
                asset_px = lonlat2tile(ds, asset_point, width, height)

                # get tile
                img = get_tile(ds,asset_px, width, height)


                def make_rectangle(u_l_point, w, h):
                    return np.array([
                        u_l_point+np.array([0,0]),  # ul
                        u_l_point+np.array([w,0]),
                        u_l_point+np.array([w,h]), 
                        u_l_point+np.array([0,h])   # lr
                        ])


                # create Polygon of created image
                img_corner = make_rectangle(asset_px, width, height)
                img_corner_lonlat = []
                for px_corner in img_corner:
                    img_corner_lonlat.append(list(pixel_to_lon_lat(ds, px_corner[0], px_corner[1])))
                img_polygon = Polygon(img_corner_lonlat)

                # get pixels of bbox;
                tower_height = 25
                tower_width = 30
                tower_point = np.array(lon_lat_to_pixel(ds, asset_point))
                tower_u_l = tower_point - np.array([tower_width // 2,tower_height // 2]) - asset_px
                bbox = make_rectangle(tower_u_l, tower_width, tower_height)
                (xmin, xmax, ymin, ymax) = (bbox[0][0], bbox[2][0],bbox[2][1], bbox[0][1])

                # Draw Bounding Box
                if draw_bb is True:
                    draw = ImageDraw.Draw(img)
                    draw.line([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)], width=1, fill='red')

                # Save Image
                filename = str(int(row.id)) # set up filename
                img.save(img_path + "_" + filename + ".png", quality=100)

                # add resulting image to dataset
                # TODO : Change (ul_x, ul_y, lr_x, lr_y) to Pascal_VOC (x_min, y_min, x_max, y_max)
                dataset = dataset.append({"filename": prefix + filename + '.png',
                                          "ul_x": xmin,
                                          "ul_y": ymax,
                                          "lr_x": xmax,
                                          "lr_y": ymin,
                                          "geometry": img_polygon}, 
                                          ignore_index=True)

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
                    dataset.to_file(img_path + "_" + "tower_examples.geojson", driver="GeoJSON")
                    # bounding_box.to_file(img_path + "tower_bbox.geojson", driver="GeoJSON")
                    return dataset


#%%
if __name__ == "__main__":
    coverage = make_polygon_list(os.path.join(os.getcwd(), "images"))
    # coverage = make_polygon_list("./")
    coverage.to_file("sierra_leone_coverage.geojson", driver="GeoJSON")
    SL_RAW_TOWERS = os.path.join(os.path.dirname(os.getcwd()), "data", "SL_raw_towers.geojson")
    make_tiled_examples(SL_RAW_TOWERS, coverage, img_path="/examples/SL", max_length=500, draw_bb=True)
    # make_examples("sierra-leone_raw_towers.geojson", coverage,
    #                       max_length=500)



# FOR DEBUGGING 
# #%%
# coverage = make_polygon_list(os.path.join(os.getcwd(), "images"))
# cov_row = coverage.loc[[0]]
# tif_fn = cov_row['filename'][0]
# tif_geo = cov_row['geometry'][0]
# print(tif_geo,tif_fn)
# # coverage.plot()
# # plt.show()

# #%%
# cov_row['geometry'].to_crs(epsg=3857).area

# #%%
# # open files
# image = tif_fn
# ds = gdal.Open(image)
# info = gdal.Info(image, format="json")
# # bands = [ds.GetRasterBand(i) for i in range(1, 4)]

# # extract relevant geographical data
# # transform = info["geoTransform"]
# # upper_left = np.array([transform[0], transform[3]])
# # pixel_size = np.array([transform[1], transform[5]])
# # print(pixel_size[0]*10000)

# #%%
# upper_left = info['cornerCoordinates']['upperLeft']
# upper_right = info['cornerCoordinates']['upperRight']
# center = info['cornerCoordinates']['center']
# # print(info['cornerCoordinates']['upperRight'])
# print(upper_left, upper_right)
# # print(pixel_size[0])
# #%%
# px_u_l = lon_lat_to_pixel(ds,upper_left)
# # px_u_r = lon_lat_to_pixel(ds,upper_right)
# # diff = tuple(x-y for x, y in zip(px_u_r, px_u_l))
# # print(px_u_r, px_u_l, diff)

# #%%
# # tif_px_size = info['size']
# # tif_px_size = np.array(tif_px_size)
# # tif_px_width = tif_px_size[0]
# # tif_px_height = tif_px_size[1]
# # print(tif_px_width, tif_px_height, tif_px_size)


# #%%
# given_px_width = 256
# given_px_height = 256
# given_size = np.array([given_px_width, given_px_height])
# # num_col_rows = tif_px_size//given_size
# # num_columns = num_col_rows[0]
# # num_rows = num_col_rows[1]
# # print('num col,rows' + str(num_col_rows))
# # print('remaining pxiels: ' + str(tif_px_size - num_col_rows*given_size))
# #%%
# px_c = lon_lat_to_pixel(ds,center)
# px_c = np.array(px_c)
# px_u_l = np.array(px_u_l)
# print(px_c, px_u_l)
# diff = px_c - px_u_l
# print(px_c, px_u_l, diff//256)

# #%%
# p_lon_lat = [-12.5705871, 8.4967127]
# p_px = lon_lat_to_pixel(ds,p_lon_lat)
# p_px = np.array(p_px)
# t_col_row = p_px//given_size
# t_px = t_col_row*given_size
# print(t_col_row, t_px)

#%%
# print(center)
# p_px = lon_lat_to_pixel(ds,p_lon_lat)
# p_lon_lat2 = pixel_to_lon_lat(ds, p_px[0], p_px[1])
# t_lat_lon = pixel_to_lon_lat(ds, t_px[0], t_px[1])
# t_lat_lon = np.array(t_lat_lon)
# p_lon_lat2 = np.array(p_lon_lat2)
# print(p_lon_lat2, p_px)
# print(lon_lat_to_pixel(ds,p_lon_lat2))