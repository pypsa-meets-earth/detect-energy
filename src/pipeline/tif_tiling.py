#%%
import os
import sys
import logging
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S')
import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm
import rasterio as rio
import rasterio.warp
import shapely


logger = logging.getLogger('maxar.repo')
# logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel(logging.DEBUG)

def generate_tiles(pixel_loc, tile_width, tile_height):
    pl_x, pl_y = pixel_loc
    tile_x, tile_y = pl_x//tile_width, pl_y//tile_height
    tile_col, tile_row = int(tile_x*tile_width), int(tile_y*tile_height)
    return (tile_col, tile_row)

def generate_windows(tile_set, tile_width, tile_height, overlap, bound_window = None):
    for tile_rc in tile_set:
        tile_col, tile_row = tile_rc
        t_window = rio.windows.Window(
            col_off=tile_col - overlap,
            row_off=tile_row - overlap,
            width=tile_width + overlap * 2,
            height=tile_height + overlap * 2)
        if bound_window is not None:
            t_window = bound_window.intersection(t_window)
        yield t_window

#%%
def tile_tif(tif_file, point_series, prefix, tile_width, tile_height, overlap, bounded):

    try:
        inds = rio.open(tif_file)
    except Exception as e:
        logger.error(e, exc_info=True)
        raise IOError
    logger.debug(f'IN width = {inds.width}, height = {inds.height}')
    if inds.count != 3:
        logger.error(f'Number of Bands is {inds.count}. Expected 3. Skipping ...')
        inds.close()
        return None

    points = point_series.to_crs(inds.crs) # Projected to raster crs
    coord_list = [(x,y) for x,y in zip(points.x , points.y)]
    pixel_loc_list = [~inds.transform * coord for coord in coord_list] # transform maps (row,col) to spatial location, ~ reverses transform

    # Generate Tile Set
    tile_set = [generate_tiles(pixel_loc, tile_width, tile_height) for pixel_loc in pixel_loc_list] # Create Tiles
    logger.debug(f'Number of Tiles created from Points = {len(tile_set)}')
    tile_set = list(dict.fromkeys(tile_set)) #remove duplicates incase there are more than two points in a tile
    logger.debug(f'Number of Tiles after removing duplicates = {len(tile_set)}')

    # Generate windows corresponding to Tile Set
    big_window = rio.windows.Window(col_off=0, row_off=0, width=inds.width, height=inds.height) if bounded else None
    window_list = generate_windows(tile_set, tile_width, tile_height, overlap, big_window)
   
    # Generate tiles as GeoTiff files      
    meta = inds.meta.copy()
    for x_window in tqdm(window_list, total=len(tile_set)):
        # File Name Example DE_X_tile_XX_XX.tif
        postfix = f'tile_{x_window.col_off}_{x_window.row_off}.tif'
        outpath = prefix + postfix
        if os.path.exists(outpath):
            logger.debug(f'{outpath} exists skipping ...')
            inds.close()
            continue
        # logger.debug(f'Writing Tif Tile with col_off =  {x_window.col_off} and row_off = {x_window.row_off}')
        # set meta
        meta['transform'] = rio.windows.transform(x_window, inds.transform)
        meta['width'], meta['height'] = x_window.width, x_window.height
        meta['driver'] = 'GTiff' # could use PNG?
        if not os.path.exists(os.path.dirname(outpath)):
            logger.error('TIF Out Path Does Not Exist')

        with rio.open(outpath, 'w', **meta) as outds:
            # b, g, r = (inds.read(k,window=window) for k in (1, 2, 3))
            # for k, arr in [(1, b), (2, g), (3, r)]:
            #   outds.write(arr, indexes=k)
            outds.write(inds.read(window=x_window))
    inds.close()





# %%
if __name__ == 'main':
    from maxar_repo import maxarImagery, maxarRepo, get_resolution
    c_dict = {'south_africa':'ZA'}
    sa_maxar_repo = maxarRepo('./maxar_repo', 'links', '/home/matin/detect_energy/osm', c_dict, '/home/matin/detect_energy')
    
    sa_repo = maxarImagery('/home/matin/detect_energy/maxar_repo/south_africa/links')

    tile_width, tile_height, overlap, bounded = 256, 256, 0, True
    
    out_path = './test_tiles'
    out_path = os.path.dirname(sa_repo.c_dir) if out_path is None else out_path
    tile_path = os.path.join(out_path, sa_repo.c_name, f'tif_tiles_{tile_width}_{tile_height}_{overlap}')
    tile_path += '_B' if bounded else ''
    os.makedirs(tile_path, exist_ok=True)
    logger.debug(f'Tiling Tifs for {sa_repo.c_name} and saving to {tile_path}')
    
    
    gdf_assets = sa_maxar_repo.get_hv_towers('/home/matin/detect_energy')
    file_point = sa_repo.get_file_point(gdf_assets)
    
    for file_number, (tif_file, point_list) in enumerate(tqdm(file_point.iteritems(),total=file_point.shape[0]),1):
            point_series = gpd.GeoSeries(point_list, crs=gdf_assets.crs)  # create GeoSeries of points
            prefix = os.path.join(tile_path, f'{sa_repo.c_name}_{file_number}_')
            if file_number == 5:
                try:
                    tile_tif(tif_file, point_series, prefix, tile_width, tile_height, overlap, bounded)
                except IOError:
                    logger.error(f'Excepted IOError for {tif_file}, skipping ...')
                    pass
                break
