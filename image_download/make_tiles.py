import gdal
from osgeo import osr
import numpy as np
from itertools import product
from gdalconst import *
from shapely.geometry import Polygon
import geopandas as gpd
import folium


def make_tiles(filename, 
               destpath, 
               width=256, 
               height=256, 
               show=False,
               make_df=True,
               max_tiles=5):
    
    """
    Takes tif file and splits it into tiles of width and height
    
    Use this function with caution, the resulting files can take
    more space than expected
    
    ----------
    Arguments:
    
    filename : (str)
        name of file to be split
        
    destpath : (str)
        resulting tiles are sent to destpath
        
    width : (int)
        number of pixels in width
        
    height : (int)
        number of pixels in height
        
    show : (bool)
        plots resulting tiles in a folium map
        
    make_df : (bool)
        creates dataframe that stores geometry of resulting tiles
        in a separate file
        
    max_tiles : (int)
        stops method after max_tiles have been created
    
    
    ----------
    Returns:
        
    resulting tif files to destpath 
    
    geopandas.GeoDataFrame storing tile geometries
    saved as .json file
    """
    
    if make_df: tiles_df = gpd.GeoDataFrame()
    
    # obtain geographic information
    srs = osr.SpatialReference()
    driver = gdal.GetDriverByName("GTiff")
    
    # load origin image
    origin = gdal.Open(filename)
    info = gdal.Info(filename, format="json")
    wkt = origin.GetProjection()
    srs.ImportFromWkt(wkt)
    
    # extract bands of origin image
    n_bands = len(info["bands"])
    bands = [origin.GetRasterBand(i) for i in range(1, n_bands+1)]

    # get origin extent in terms of pixel numbers
    cols = origin.RasterXSize
    rows = origin.RasterYSize
    x_steps = np.arange(0, cols//width) * width
    y_steps = np.arange(0, rows//height) * height
    
    # get origin extent in terms of geometry and pixel sizes
    tf = origin.GetGeoTransform()
    x_origin = tf[0]
    y_origin = tf[3]
    pixel_width = tf[1]
    pixel_height = -tf[5]
    shifts = np.array([width*pixel_width, height*pixel_height])
    
    for num, (x, y) in enumerate(product(x_steps, y_steps)):
        
        # each x, y combination refers to the upper left pixel of each tile
        x, y = int(x), int(y)
        
        curr_bands = [band.ReadAsArray(x, y, width, height) for band in bands]
        curr_coords = np.array([x_origin + x*pixel_width, y_origin + y*pixel_height])
        as_polygon = Polygon([
                             [curr_coords[1], curr_coords[0]],
                             [curr_coords[1]+shifts[1], curr_coords[0]],
                             [curr_coords[1]+shifts[1], curr_coords[0]+shifts[0]],
                             [curr_coords[1], curr_coords[0]+shifts[0]]
                             ])
            
        curr_transform = (curr_coords[1], tf[1], tf[2], curr_coords[0], tf[4], tf[5])
        
        # set outside file
        out_file = "-".join(map(str, curr_coords))
        out_file = destpath + out_file.replace(".", "_") + ".tif"
        
        # initiate output tif
        dst = driver.Create(out_file, 
                       width,
                       height,
                       n_bands,
                       gdal.GDT_Float32
                       )
        
        # transfer image data to bands
        colors = [GCI_RedBand, GCI_GreenBand, GCI_BlueBand] if n_bands == 3 else [GCI_GrayIndex]
        for i, band, color in zip(range(1, n_bands+1), curr_bands, colors):            
            
            dst.GetRasterBand(i).WriteArray(band)            
            dst.GetRasterBand(i).SetColorInterpretation(color)
        
        # transfer geographic data
        dst.SetGeoTransform(curr_transform)
        dst.SetProjection(srs.ExportToWkt())
        dst = None
        
        if make_df: 
            tiles_df = tiles_df.append({"geometry": as_polygon}, ignore_index=True)
            with open(destpath + 'tile_geometries.json', 'w') as f:
                f.write(tiles_df.to_json())
            
        if num == max_tiles: break

            
def show_tiles(gdf):
    """
    Shows tile geometry in a folium map
    
    ---------
    Arguments:
        
    gdf : (str) or (geopandas.GeoDataFrame)
        either path to json file of gdf or the gdf itself
    """
    
    if isinstance(gdf, str): gdf = gpd.read_file(gdf)
    
    center = np.array(gdf.iloc[0].geometry.centroid).tolist()
    center.reverse()
    
    m = folium.Map(location=center,
                   zoom_start=11
                   )
    
    for idx, row in gdf.iterrows():
        
        sim_geo = gpd.GeoSeries(row['geometry']).simplify(tolerance=0.001)
        geo_j = sim_geo.to_json()
        geo_j = folium.GeoJson(data=geo_j, 
                                style_function=lambda x: {'fillColor': 'orange'})
        geo_j.add_to(m)
    
            
    return m
            

if __name__ == "__main__":
    make_tiles("./images/sierra_leone_001.tif", "./tiles/SL_")         
    m = show_tiles("./tiles/SLtile_geometries.json")
    m