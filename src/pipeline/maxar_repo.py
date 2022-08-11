import os
import logging
import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm
import rasterio as rio
import rasterio.warp
import shapely

from osm_towers import add_geom_towers


logger = logging.getLogger('maxar.repo')


def tile_tif(tif_file, point_series, prefix, tile_width, tile_height, overlap, bounded):
    with rio.open(tif_file) as inds: 
        logger.debug(f'IN width = {inds.width}, height = {inds.height}')
        if inds.count != 3:
            logger.error(f'Number of Bands is {inds.count}. Expected 3')
            return None

        points = point_series.to_crs(inds.crs) # Projected to raster crs
        coord_list = [(x,y) for x,y in zip(points.x , points.y)]

        # Generate requred tiles
        tile_set=[]
        for coord in coord_list:
            pixel_loc = ~inds.transform * coord # transform maps (row,col) to spatial location, ~ reverses transform
            pl_x, pl_y = pixel_loc
            tile_x, tile_y = pl_x//tile_width, pl_y//tile_height
            tile_col, tile_row = int(tile_x*tile_width), int(tile_y*tile_height)
            tile_set.append((tile_col, tile_row))

        tile_set = list(dict.fromkeys(tile_set)) #remove duplicates incase there are more than two points in a tile
        yield tile_set # return the tiles that will be created

        # Generate windows corresponding to tiles
        ncols, nrows = inds.width, inds.height
        big_window = rio.windows.Window(col_off=0, row_off=0, width=ncols, height=nrows)
        window_list = []
        for tile_rc in tile_set:
            tile_col, tile_row = tile_rc
            t_window = rio.windows.Window(
                col_off=tile_col - overlap,
                row_off=tile_row - overlap,
                width=tile_width + overlap * 2,
                height=tile_height + overlap * 2)
            if bounded:
                t_window = big_window.intersection(t_window)
            window_list.append(t_window)
        
        # Generate tiles as GeoTiff files      
        meta = inds.meta.copy()
        for x_window in window_list:
            # set meta
            meta['transform'] = rio.windows.transform(x_window, inds.transform)
            meta['width'], meta['height'] = x_window.width, x_window.height
            meta['driver'] = 'GTiff' # could use PNG?
            # TODO: find file_number by listdir tif files, also use country code instead of name
            # File Name Example 
            postfix = f'tile_{x_window.col_off}_{x_window.row_off}'
            outpath = prefix + postfix
            with rio.open(outpath, 'w', **meta) as outds:
                # b, g, r = (inds.read(k,window=window) for k in (1, 2, 3))
                # for k, arr in [(1, b), (2, g), (3, r)]:
                #   outds.write(arr, indexes=k)
                outds.write(inds.read(window=x_window))


def get_utm_from_wgs(w_lon, s_lat, e_lon, n_lat):
    from pyproj import CRS
    from pyproj.aoi import AreaOfInterest
    from pyproj.database import query_utm_crs_info
    
    utm_crs_list = query_utm_crs_info(
        datum_name="WGS 84",
        area_of_interest=AreaOfInterest(
            west_lon_degree=w_lon,
            south_lat_degree=s_lat,
            east_lon_degree=e_lon,
            north_lat_degree=n_lat,
        ),
    )
    if len(utm_crs_list) == 0:
        logger.error('probably already in utm')
        return None
    utm_crs = CRS.from_epsg(utm_crs_list[0].code) # just return the first one for now
    return utm_crs.to_string()
    

class maxarImagery:
    def __init__(self, tif_dir):
        self.tif_dir = tif_dir
        self.tif_paths = [os.path.join(self.tif_dir, filename) for filename in os.listdir(self.tif_dir) if filename.endswith(".tif")]
        self.coverage_path = os.path.join(self.tif_dir, "coverage.geojson")
        self.c_name = os.path.basename(os.path.dirname(tif_dir))
        self.c_dir = os.path.dirname(tif_dir)


    def get_info(self):
        for tif_file in self.tif_paths[0:1]:
            with rio.open(tif_file) as s:
                logger.info('\n'.join(['',
                    f"CRS = {s.crs}",
                    f"Width = {s.width}, Height = {s.height}",
                    f"{s.transform}",
                    tuple(s.transform),
                    f"Transform (GDAL) {s.transform.to_gdal()}",
                    f"Transform (Shapely) {s.transform.to_shapely()}",
                    s.tags(),
                    f"Meta = {s.meta}",
                    f"Driver = {s.driver}",
                    # f"Dataset Mask = {s.dataset_mask()}",
                    f"Bounds = {s.bounds}",
                    f"Number of Bands = {s.count}"
                ]))

    def get_resolution(self):
        for tif_file in self.tif_paths[0:1]:
            with rio.open(tif_file) as src:
                utm_dst_crs = get_utm_from_wgs(*src.bounds) if src.crs.is_geographic else src.crs
                transform, width, height = rio.warp.calculate_default_transform(src.crs, utm_dst_crs, src.width, src.height, *src.bounds)
                logger.info(f'x_res = {transform.a}, y_res = {-transform.e}')
                return(transform.a, -transform.e)
    
    def get_crs_code(self):
        for tif_file in self.tif_paths[0:1]: # CRS for all files in one CC are the same
            with rio.open(tif_file) as src:
                crs = src.crs
                # crs_code = str(crs.code)
                crs_code = str(crs['init']).split(':')[-1]
                logger.info(f'{crs_code}')
            return crs_code
    
    def get_tif_bounds(tif_file_path, dst_crs={'init': 'epsg:4326'}):
        if dst_crs is None:
            reproject = False
        else:
            reproject = True

        with rio.open(tif_file_path) as src:
            bounds = src.bounds
            raster_crs = src.crs
            if raster_crs != dst_crs and reproject is True:
                logger.debug(f"reprojecting crs from {src.crs} to {dst_crs}")
                bounds = rasterio.warp.transform_bounds(src.crs, dst_crs, *bounds)
                raster_crs = dst_crs
            geom = shapely.geometry.box(*bounds)
        return geom, raster_crs

    def generate_coverage(self, save=True):
        coverage = gpd.GeoDataFrame({"filename": [], "geometry": []})
        for tif_file in tqdm(self.tif_paths):
            poly, crs = self.get_tif_bounds(tif_file, dst_crs=None)
            coverage = coverage.append({"filename": tif_file, "geometry": poly}, ignore_index=True)
        coverage = coverage.set_crs(epsg=self.get_crs_code())
        if save:
            coverage.to_file(self.coverage_path, driver='GeoJSON')
        return coverage

    def get_coverage(self, cache=True):
        if not os.path.exists(self.coverage_path):
            self.generate_coverage(save=True)
        
        coverage = gpd.read_file(self.coverage_path)
        return coverage

    def tile_tif(self, final_assets):
        gc = self.get_coverage()
        joined_assets = final_assets.sjoin(gc, how='inner')
        file_point = joined_assets.groupby('filename')['geometry'].apply(list)
        tile_width, tile_height, overlap, bounded = 256, 256, 0, True
        
        for file_number, (tif_file, point_list) in enumerate(tqdm(file_point.iteritems(),total=file_point.shape[0]),1):
                point_series = gpd.GeoSeries(point_list, crs=4326)  # create GeoSeries of points
                prefix = os.path.join(self.c_dir, f'tif_tiles_{tile_width}_{tile_height}_{overlap}')
                prefix += '_B' if bounded else ''
                prefix = os.path.join(prefix, f'{self.c_name}_{file_number}')
                tile_set = tile_tif(tif_file, point_series, prefix)



class maxarRepo:
    def __init__(self, repo_path, osm_path, country_dict):
        self.repo_path = '/mnt/gdrive/maxar'
        self.osm_path = '/mnt/gdrive/osm'
        self.country_dict = { 
            # Dict of all currently available imagery
            'australia': 'AU',
            'bangladesh': 'BD',
            'chad': 'TD',
            'drc': 'CD',
            'ghana': 'GH',
            'malawi': 'MW',
            'sierra_leone': 'SL',
            'california': 'US-CA',
            'texas': 'US-TX',
            'brazil': 'BR',
            'south_africa':'ZA',
            'germany': 'DE',
            'philippines': 'PH'
            }
        if country_dict is not None:
            self.country_dict = country_dict
        else:
            self.country_dirs = os.listdir(self.repo_path)
            logger.info (print('missing regions in all_dict', set(self.country_dirs) - set(self.country_dict.keys())))
        
        self.tif_dirs = [os.path.join(self.repo_path, c_dir,'raw') for c_dir in self.country_dict.keys()]

    def get_hv_towers(self):
            # TODO: Move this function somewhere else
            # check if assets available
            for c_dir in self.country_dict.keys():
                if not os.path.exists(os.path.join(self.osm_path,f'{self.country_dict[c_dir]}_raw_towers.geojson')):
                    logger.error(f'asset not found for {c_dir} with code {self.country_dict[c_dir]}')
            
            asset_paths = [os.path.join(self.osm_path,f'{self.country_dict[c_dir]}_raw_towers.geojson') for c_dir in self.country_dict.keys()]
            tower_paths = [os.path.join(self.osm_path,f'{self.country_dict[c_dir]}_raw_towers.csv') for c_dir in self.country_dict.keys()]
            line_paths = [os.path.join(self.osm_path,f'{self.country_dict[c_dir]}_raw_lines.csv') for c_dir in self.country_dict.keys()]

            # Combine Tower GeoJSON to GeoPandas DF
            assets_all = gpd.GeoDataFrame()
            for asset_path in asset_paths:
                asset = gpd.read_file(asset_path)
                assets_all = assets_all.append(asset, ignore_index = True)
            assets_all = assets_all.set_crs(epsg='4326')

            # Combine Tower CSV to Pandas DF
            tower_all = pd.concat(map(pd.read_csv, tower_paths))

            # Combine Lines CSV to Pandas DF
            line_all = pd.concat(map(pd.read_csv, line_paths))

            gdf_hv_t = add_geom_towers(assets_all, tower_all, line_all, hv_thresh = 100000)

            return gdf_hv_t

    def get_crs_dict(self):
        crs_dict = {}
        for tif_dir in self.tif_dirs:
            m_sat = maxarImagery(tif_dir)
            crs_code = m_sat.get_crs_code()
            crs_dict[m_sat.c_name] = str(crs_code)
        logger.info(crs_dict)
        return crs_dict



    def get_repo_resolution(self):
        # Check Resolution of Dataset
        for tif_dir in self.tif_dirs:
            m_sat = maxarImagery(tif_dir)
            x_res, y_res = m_sat.get_resolution
            logger.info(f'country: {m_sat.c_name}, x_res = {x_res}, y_res = {y_res}')


    def get_total_coverage(self):
        # coverage_paths = [os.path.join(ds_path,c_dir,'coverage.geojson') for c_dir in all_dict.keys()]
        total_coverage = gpd.GeoDataFrame()
        for tif_dir in self.tif_dirs:
            m_sat = maxarImagery(tif_dir)
            coverage = m_sat.get_coverage()
            coverage = coverage.to_crs(epsg='4326') # standardize crs
            total_coverage = total_coverage.append(coverage, ignore_index=True)
        return total_coverage



            








# Dict for experiments (subset of all_dict)
c_dict = {
    'australia': 'AU',
    'bangladesh': 'BD',
    # 'chad': 'TD',
    # 'drc': 'CD',
    'ghana': 'GH',
    # 'malawi': 'MW',
    # 'sierra_leone': 'SL',
    # 'california': 'US-CA',
    # 'texas': 'US-TX',
    # 'brazil': 'BR',
    'south_africa':'ZA',
    # 'germany': 'DE',
    # 'philippines': 'PH'
    }


myMaxar = maxarRepo('/mnt/gdrive/maxar', '/mnt/gdrive/osm', c_dict)

# print(myMaxar.get_hv_towers())
print(myMaxar.get_crs_dict())

# ghana = 

# sudo rclone mount mygrdrive:/PyPSA_Africa_images /mnt/gdrive --config=/home/matin/.config/rclone/rclone.conf --allow-other --drive-shared-with-me

# # Check if asset files exist (run osm-extractor if not found)
# for c_dir in all_dict.keys():
#   if not os.path.exists(os.path.join(osm_path,f'{all_dict[c_dir]}_raw_towers.geojson')):
#     print(f'asset not found for {c_dir} with code {all_dict[c_dir]}')


# Visualize Coverage

# import folium
# m = folium.Map(zoom_start=10, tiles='CartoDB positron')

# for _, r in total_coverage.iterrows():
#     # Without simplifying the representation of each borough,
#     # the map might not be displayed
#     # sim_geo = gpd.GeoSeries(r['geometry']).simplify(tolerance=0.001)
#     sim_geo = gpd.GeoSeries(r['geometry'])
#     geo_j = sim_geo.to_json()
#     geo_j = folium.GeoJson(data=geo_j,
#                            style_function=lambda x: {'fillColor': 'orange'})
#     # folium.Popup(r['BoroName']).add_to(geo_j)
#     geo_j.add_to(m)
# m